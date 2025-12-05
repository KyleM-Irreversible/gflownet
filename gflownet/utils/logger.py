import re
import os
import sys
import time
import threading
import queue
from datetime import datetime
from pathlib import Path
from typing import Union
import logging

import matplotlib.pyplot as plt
import numpy as np
import torch
from numpy import array
from omegaconf import OmegaConf
from torchtyping import TensorType



class AsyncMLFlowLogger:
    def __init__(self, run_name, flush_interval=2):
        import mlflow
        self.mlflow = mlflow
        self.run_name = run_name
        self.flush_interval = flush_interval
        self.queue = queue.Queue()
        self.stop_flag = False
        self.thread = threading.Thread(target=self.worker, daemon=True)
        self.thread.start()

        # dynamic logging control
        self.target_queue = 50
        self._block_steps = set()
        self._last_seen_step = -1


    def should_log(self, step):
        """Keep track of the time between steps and adust logging frequency to skip steps to keep pace. If a step value is skipped, all subsequent calls with the same step should also be skipped."""
        if step is None:
            raise ValueError("Step must be provided for logging.")
        
        if step in self._block_steps:
            return False
        
        if step <=1:
            return True
                
        if step > self._last_seen_step:
            self._last_seen_step = step

            current_queue_size = self.queue.qsize()
            if current_queue_size > self.target_queue:
                self._block_steps.add(step)
                return False
        
        return True
        

    
    def clean(self, s):
        return re.sub(r"[^a-zA-Z0-9_\-\. :/]", "", s)


    def worker(self):
        with self.mlflow.start_run(run_name=self.run_name) as self.mlflow_run:
            while not self.stop_flag:
                try:
                    func, args, kwargs = self.queue.get(timeout=self.flush_interval)
                    func(*args, **kwargs)
                except queue.Empty:
                    pass

    def log_metric(self, key, value, step=None):
        if self.should_log(step):
            key = self.clean(key)
            self.queue.put((self.mlflow.log_metric, (key, value), {"step": step}))

    def log_param(self, key, value):
        key = self.clean(key)
        self.queue.put((self.mlflow.log_param, (key, value), {}))

    def close(self):
        self.stop_flag = True
        self.thread.join()
    
    def catch_up(self):
        return 
        """Block until the queue is empty (flush pending logs)."""
        while not self.queue.empty():
            try:
                func, args, kwargs = self.queue.get_nowait()
                func(*args, **kwargs)
            except queue.Empty:
                break


class Logger:
    """
    Utils functions to compute and handle the statistics (saving them or send to
    wandb). It can be passed on to querier, gfn, proxy, ... to get the
    statistics of training of the generated data at real time

    Parameters
    ----------
    run_name : str
        Name of the run. By default it is None. If run_name is None and run_name_date
        and run_name_job are both False, then a random name will be assigned by wandb.
    run_name_date : bool
        Whether the date (and time) should be included in the run name. True by
        default.
    run_name_job : bool
        Whether the job ID should be included in the run name. True by default.
    progressbar : dict
        A dictionary of configuration parameters related to the progress bar, namely:
            - skip : bool
                If True, the progress bar is not displayed during training. False by
                default.
            - n_iters_mean : int
                The number of past iterations to take into account to compute averages
                of a metric, for example the loss. 100 by default.

    """

    def __init__(
        self,
        config: dict,
        do: dict,
        project_name: str,
        logdir: dict,
        lightweight: bool,
        debug: bool,
        run_name=None,
        run_name_date: bool = True,
        run_name_job: bool = True,
        run_id: str = None,
        tags: list = None,
        context: str = "0",
        notes: str = None,
        entity: str = None,
        progressbar: dict = {"skip": False, "n_iters_mean": 100},
        is_resumed: bool = False,
    ):
        self.config = config
        self.do = do
        self.do.times = self.do.times and self.do.online
        slurm_job_id = os.environ.get("SLURM_JOB_ID")

        # Determine run name
        if run_name is None:
            run_name = ""
        if run_name_job and slurm_job_id is not None:
            run_name = f"{run_name} {slurm_job_id}"
        if run_name_date:
            date_time = datetime.today().strftime("%d/%m-%H:%M:%S")
            run_name = f"{run_name} {date_time}"

        # MLflow support
        self.mlflow = None
        self.mlflow_run = None
        if hasattr(self.do, "mlflow") and self.do.mlflow:
            try:
                import mlflow
                self.mlflow = mlflow
                self.mlflow.set_tracking_uri(self.do.mlflow.get("tracking_uri", ""))
                self.mlflow.set_experiment(project_name)
                
                self.mlflow_logger = AsyncMLFlowLogger(run_name=run_name)

                # Log config as params
                for k, v in config.items():
                    if isinstance(v, dict):
                        for kk, vv in v.items():
                            self.mlflow_logger.log_param(f"{k}.{kk}", vv)
                    else:
                        self.mlflow_logger.log_param(k, v)
            except Exception as e:
                print(f"MLflow init failed: {e}")
                self.mlflow = None
                self.mlflow_run = None

        if self.do.online:
            import wandb

            self.wandb = wandb
            wandb_config = OmegaConf.to_container(
                config, resolve=True, throw_on_missing=True
            )
            if slurm_job_id:
                wandb_config["slurm_job_id"] = slurm_job_id
            if run_id is not None:
                # Resume run
                self.run = self.wandb.init(
                    id=run_id,
                    project=project_name,
                    entity=entity,
                    resume="allow",
                )
            else:
                self.run = self.wandb.init(
                    config=wandb_config,
                    project=project_name,
                    name=run_name,
                    notes=notes,
                    entity=entity,
                    resume="allow",
                )
        else:
            self.wandb = None
            self.run = None

        self.add_tags(tags)
        self.context = context
        self.progressbar = progressbar
        self.loss_memory = []
        self.lightweight = lightweight
        self.debug = debug
        self.is_resumed = is_resumed
        # Log directory
        if "path" in logdir:
            self.logdir = Path(logdir.path)
        else:
            self.logdir = Path(logdir.root)
        if self.is_resumed:
            if self.debug:
                print(f"Run is resumed and will log into directory {self.logdir}")
        elif not self.logdir.exists() or logdir.overwrite:
            self.logdir.mkdir(parents=True, exist_ok=True)
        else:
            print(f"logdir {logdir} already exists! - Ending run...")
            sys.exit(1)
        # Checkpoints directory
        self.ckpts_dir = self.logdir / logdir.ckpts
        self.ckpts_dir.mkdir(parents=True, exist_ok=True)
        # Data directory
        self.datadir = self.logdir / "data"
        self.datadir.mkdir(parents=True, exist_ok=True)
        # Write wandb URL
        self.write_url_file()

    def write_url_file(self):
        if self.wandb is not None:
            self.url = self.wandb.run.get_url()
            if self.url:
                with open(self.logdir / "wandb.url", "w") as f:
                    f.write(self.url + "\n")

    def add_tags(self, tags: list):
        if not self.do.online:
            return
        if tags:
            self.run.tags = self.run.tags + tags

    def set_context(self, context: int):
        self.context = str(context)

    def progressbar_update(
        self, pbar, loss, rewards, jsd, use_context=True, n_mean=100
    ):
        if self.progressbar["skip"]:
            return
        if len(self.loss_memory) < self.progressbar["n_iters_mean"]:
            self.loss_memory.append(loss)
        else:
            self.loss_memory = self.loss_memory[1:] + [loss]
        description = "Loss: {:.4f} | Mean rewards: {:.2f} | JSD: {:.4f}".format(
            np.mean(self.loss_memory), np.mean(rewards), jsd
        )
        pbar.update(1)
        pbar.set_description(description)

    def log_histogram(self, key, value, step, use_context=True):
        # Log to wandb
        if self.do.online:
            if use_context:
                key = self.context + "/" + key
            fig = plt.figure()
            plt.hist(value)
            plt.title(key)
            plt.ylabel("Frequency")
            plt.xlabel(key)
            figimg = self.wandb.Image(fig)
            self.wandb.log({key: figimg}, step)
            plt.close(fig)
        # Log to MLflow
        if self.mlflow is not None:
            fig = plt.figure()
            plt.hist(value)
            plt.title(key)
            plt.ylabel("Frequency")
            plt.xlabel(key)
            img_path = self.logdir / f"{key}_hist_step{step}.png"
            fig.savefig(img_path)
            self.mlflow.log_artifact(str(img_path))
            plt.close(fig)

    def log_plots(self, figs: Union[dict, list], step, use_context=True):
        keys = None
        if isinstance(figs, dict):
            keys = list(figs.keys())
            figs = list(figs.values())
        else:
            assert isinstance(figs, list), "figs must be a list or a dict"
            keys = [f"Figure {i} at step {step}" for i in range(len(figs))]

        # Log to wandb
        if self.do.online:
            for key, fig in zip(keys, figs):
                log_key = self.context + "/" + key if use_context else key
                if fig is not None:
                    figimg = self.wandb.Image(fig)
                    self.wandb.log({log_key: figimg}, step)
            self.close_figs(figs)
        # Log to MLflow
        if self.mlflow is not None:
            for key, fig in zip(keys, figs):
                log_key = self.context + "/" + key if use_context else key
                if fig is not None:
                    img_path = self.logdir / f"{log_key}_step{step}.png"
                    fig.savefig(img_path)
                    self.mlflow.log_artifact(str(img_path))
            self.close_figs(figs)

    def close_figs(self, figs: list):
        for fig in figs:
            if fig is not None:
                plt.close(fig)

    def log_rewards_and_scores(
        self,
        rewards: TensorType["n_samples"],
        logrewards: TensorType["n_samples"],
        scores: TensorType["n_samples"],
        step: int,
        prefix: str,
        use_context: bool = True,
    ):
        """
        Logs the rewards, log-rewards and proxy scores passed as arguments.
        """
        metrics = {
            f"{prefix} rewards mean": rewards.mean(),
            f"{prefix} rewards max": rewards.max(),
            f"{prefix} logrewards mean": logrewards.mean(),
            f"{prefix} logrewards max": logrewards.max(),
        }
        if scores is not None:
            metrics.update(
                {
                    f"{prefix} scores mean": scores.mean(),
                    f"{prefix} scores min": scores.min(),
                    f"{prefix} scores max": scores.max(),
                }
            )
        self.log_metrics(metrics, step=step, use_context=use_context)

    def log_metrics(
        self,
        metrics: dict,
        step: int,
        use_context: bool = True,
    ):
        
        """
        Logs metrics to wandb and/or MLflow.
        """


        # Log to wandb
        if self.do.online:
            for key, value in metrics.items():
                log_key = self.context + "/" + key if use_context else key
                if value is None:
                    continue
                self.wandb.log({log_key: value}, step=step)
        # Log to MLflow
        if self.mlflow is not None:
            if step % self.do.mlflow.get('sync_interval', 1000) == 0 and step > 0: 
                #Allow the mlflow logger to catch up every sync_interval steps
                self.mlflow_logger.catch_up()
                 
            for key, value in metrics.items():
                log_key = self.context + "/" + key if use_context else key
                if value is None:
                    continue
                try:
                    self.mlflow_logger.log_metric(log_key, float(value), step=step)
                except Exception as e:
                    if self.debug:
                        print(f"MLflow log_metric failed for {log_key}: {e}")

    def log_summary(self, summary: dict):
        # Log to wandb
        if self.do.online:
            self.run.summary.update(summary)
        # Log to MLflow
        if self.mlflow is not None:
            for k, v in summary.items():
                try:
                    self.mlflow_logger.log_param(f"summary.{k}", v)
                except Exception as e:
                    if self.debug:
                        print(f"MLflow log_param failed for summary.{k}: {e}")

    def save_checkpoint(
        self,
        forward_policy,
        backward_policy,
        state_flow,
        logZ,
        optimizer,
        buffer,
        step: int,
        final: bool = False,
    ):
        if final:
            ckpt_id = "final"
            if self.debug:
                print("Saving final checkpoint in:")
        else:
            ckpt_id = "iter_{:06d}".format(step)
            if self.debug:
                print(f"Saving checkpoint of step {step} in:")

        ckpt_path = self.ckpts_dir / (ckpt_id + ".ckpt")
        if self.debug:
            print(f"\t{ckpt_path}")

        # Forward model
        if forward_policy.is_model:
            forward_ckpt = forward_policy.model.state_dict()
        else:
            forward_ckpt = None

        # Backward model
        if backward_policy and backward_policy.is_model:
            backward_ckpt = backward_policy.model.state_dict()
        else:
            backward_ckpt = None

        # State flow model
        if state_flow:
            state_flow_ckpt = state_flow.model.state_dict()
        else:
            state_flow_ckpt = None

        # LogZ
        if isinstance(logZ, torch.nn.Parameter) and logZ.requires_grad:
            logZ_ckpt = logZ.detach().cpu()
        else:
            logZ_ckpt = None

        # Buffer
        buffer_ckpt = {
            "train": None,
            "test": None,
            "replay": None,
        }
        if hasattr(buffer, "train") and buffer.train is not None:
            if hasattr(buffer.train_config, "pkl") and buffer.train_config.pkl:
                buffer_ckpt["train"] = str(buffer.train_config.pkl)
        if hasattr(buffer, "test") and buffer.test is not None:
            if hasattr(buffer.test_config, "pkl") and buffer.test_config.pkl:
                buffer_ckpt["test"] = str(buffer.test_config.pkl)
        if hasattr(buffer, "replay") and len(buffer.replay) > 0:
            if hasattr(buffer, "replay_csv") and buffer.replay_csv is not None:
                buffer_ckpt["replay"] = str(buffer.replay_csv)

        # WandB run ID
        if self.do.online:
            run_id_ckpt = self.run.id
        else:
            run_id_ckpt = None

        checkpoint = {
            "step": step,
            "forward": forward_ckpt,
            "backward": backward_ckpt,
            "state_flow": state_flow_ckpt,
            "logZ": logZ_ckpt,
            "optimizer": optimizer.state_dict(),
            "buffer": buffer_ckpt,
            "run_id": run_id_ckpt,
        }
        torch.save(checkpoint, ckpt_path)
        # Log checkpoint to MLflow
        if self.mlflow is not None:
            try:
                self.mlflow.log_artifact(str(ckpt_path))
            except Exception as e:
                if self.debug:
                    print(f"MLflow log_artifact failed: {e}")

    def log_time(self, times: dict, use_context: bool):
        if self.do.times:
            times = {"time_{}".format(k): v for k, v in times.items()}
            self.log_metrics(times, use_context=use_context)

    def end(self):
        if self.do.online:
            self.wandb.finish()
        if self.mlflow is not None:
            logging.info("Closing MLflow logger...")
            self.mlflow_logger.close()
            try:
                self.mlflow.end_run()
            except Exception as e:
                if self.debug:
                    print(f"MLflow end_run failed: {e}")

