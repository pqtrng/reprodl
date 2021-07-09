import logging
from pathlib import Path

import hydra
from crontab import CronTab
from omegaconf import DictConfig
from omegaconf import OmegaConf

logger = logging.getLogger(__name__)


@hydra.main(config_path="configs", config_name="default")
def schedule(cfg: DictConfig):
    """Scheduler to run cronjobs.

    Args:
        cfg (DictConfig): config for running
    """
    logger.info(OmegaConf.to_yaml(cfg))

    path = Path(hydra.utils.get_original_cwd())

    data_crontab = CronTab(cfg.cron.username)

    print("Current data jobs")
    for job in data_crontab:
        print(job)
        if cfg.cron.clean:
            data_crontab.remove(job)
            print(f"Remove {job} from job list.")

    if not cfg.cron.stop:
        for file_name, cron_params in cfg.cron.py_cmds.items():
            cwd_cmd = "cd " + str(path)

            py_cmd = f"{cfg.cron.python_path} python {str(path)}/{file_name}.py"
            cmd = f"{cwd_cmd}; {py_cmd}"
            print(f"\tCurrent command:\n\t{cmd}")

            # Check if command is running
            for i in data_crontab.find_command(cmd):
                print(str(i))
                if cmd in str(i):
                    time_info = str(i).split(cmd)[0].strip()
                    print(time_info)

                    print("\tEXISTS")
                print("\tNO")

            job = data_crontab.new(command=cmd)

            for key_name, key_value in cron_params.items():
                for k_job, v_job in key_value.items():
                    print(f"\t\t{key_name} - {k_job} - {v_job}")
                    getattr(getattr(job, str(key_name)), str(k_job))(str(v_job))

    data_crontab.write()


if __name__ == "__main__":
    schedule()
