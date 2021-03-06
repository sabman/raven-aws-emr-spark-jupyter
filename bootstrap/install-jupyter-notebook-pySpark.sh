# Copyright (c) 2015 Giving.com, trading as JustGiving. Licensed under the Apache License (v2.0).
#!/bin/bash
set -x -e

#Installing iPython Notebook
if grep isMaster /mnt/var/lib/info/instance.json | grep true;
then
cd /home/hadoop
sudo pip install virtualenv
mkdir IPythonNB
cd IPythonNB
`which virtualenv` -p /usr/bin/python2.7 venv
source venv/bin/activate

#Install ipython and dependency
pip install "ipython[notebook]"
pip install requests numpy
pip install matplotlib

#Create profile
ipython profile create default

#Run on master /slave based on configuration
echo "c = get_config()" >  /home/hadoop/.ipython/profile_default/ipython_notebook_config.py
echo "c.NotebookApp.ip = '*'" >>  /home/hadoop/.ipython/profile_default/ipython_notebook_config.py
echo "c.NotebookApp.open_browser = False"  >>  /home/hadoop/.ipython/profile_default/ipython_notebook_config.py
echo "c.NotebookApp.port = 8192" >>  /home/hadoop/.ipython/profile_default/ipython_notebook_config.py
echo "c.NotebookApp.password = u'sha1:14445163bd5a:97a0f8d56520c8c4f342a0145425fc362150adee'" >> /home/hadoop/.ipython/profile_default/ipython_notebook_config.py

#starting ipython notebook with pyspark interactive support.
export IPYTHON_HOME=/home/hadoop/IPythonNB/venv/
export PATH=$PATH:$IPYTHON_HOME/bin
export IPYTHON_OPTS="notebook --no-browser"
export MASTER=yarn-client
export PYSPARK_PATH=`which pyspark`
nohup $PYSPARK_PATH --master yarn-client > /mnt/var/log/python_notebook.log &

fi
