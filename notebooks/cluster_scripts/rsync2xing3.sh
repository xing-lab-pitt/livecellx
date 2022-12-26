#!/bin/sh
rsync -rav --progress --ignore-existing ./lightning_logs/version_* ken67@xing-lab-3.csb.pitt.edu:/home/ken67/LiveCellTracker-dev/notebooks/lightning_logs/

rsync -rav --progress --size-only ken67@xing-lab-3.csb.pitt.edu:/home/ken67/LiveCellTracker-dev/notebooks/lightning_logs/ ./lightning_logs/