#!/bin/sh

# to workstation 3
rsync -rav --progress --ignore-existing ./lightning_logs/version_* ken67@xing-lab-3.csb.pitt.edu:/home/ken67/LiveCellTracker-dev/notebooks/lightning_logs/

# to pitt cluster
rsync -rav --progress --ignore-existing ./lightning_logs/version_* ken67@cluster.csb.pitt.edu:/net/capricorn/home/xing/ken67/LiveCellTracker-dev/notebooks/lightning_logs/

# from workstation 3
rsync -rav --progress --ignore-existing ken67@xing-lab-3.csb.pitt.edu:/home/ken67/LiveCellTracker-dev/notebooks/lightning_logs/ ./lightning_logs/

# from pitt cluster
rsync -rav --progress --ignore-existing ken67@cluster.csb.pitt.edu:/net/capricorn/home/xing/ken67/LiveCellTracker-dev/notebooks/lightning_logs/ ./lightning_logs/