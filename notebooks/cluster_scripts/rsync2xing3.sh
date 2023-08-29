#!/bin/sh

# to workstation 3
rsync -rav --progress --ignore-existing ./lightning_logs/version_* ken67@xing-lab-3.csb.pitt.edu:/home/ken67/LiveCellTracker-dev/notebooks/lightning_logs/

# to pitt cluster
rsync -rav --progress --ignore-existing ./lightning_logs/version_* ken67@cluster.csb.pitt.edu:/net/capricorn/home/xing/ken67/LiveCellTracker-dev/notebooks/lightning_logs/

# from workstation 3
rsync -rav --progress --ignore-existing ken67@xing-lab-3.csb.pitt.edu:/home/ken67/LiveCellTracker-dev/notebooks/lightning_logs/ ./lightning_logs/

# from pitt cluster
rsync -rav --progress --ignore-existing ken67@cluster.csb.pitt.edu:/net/capricorn/home/xing/ken67/LiveCellTracker-dev/notebooks/lightning_logs/ ./lightning_logs/

# to pitt cluster
rsync -rav --progress --ignore-existing ./notebook_results/a549_ccp_vim/*_data_v11 ken67@cluster.csb.pitt.edu:/net/capricorn/home/xing/ken67/LiveCellTracker-dev/notebooks/notebook_results/a549_ccp_vim


rsync -rav --progress --ignore-existing ken67@cluster.csb.pitt.edu:/net/capricorn/home/xing/ken67/.aws ~/.aws

rsync -rav --progress --ignore-existing ./notebook_results/mmaction_train_data* ken67@cluster.csb.pitt.edu:/net/capricorn/home/xing/ken67/LiveCellTracker-dev/notebooks/notebook_results

rsync -rav --progress --ignore-existing ken67@cluster.csb.pitt.edu:/net/capricorn/home/xing/ken67/LiveCellTracker-dev/notebooks/scripts/mmdetection_classify/work_dirs/* ./scripts/mmdetection_classify/work_dirs/