Copyright (c) 2015 Giving.com, trading as JustGiving.

Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance with the License. A copy of the License is located in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

# Running Jupyter and pySpark on an Apache Spark Cluster on Elastic MapReduce (Amazon EMR)

## Introduction
This guide will allow any data scientist or analyst to run Python code on an [Apcahe Spark](https://spark.apache.org/) cluster in a familiar [Jupyter](https://jupyter.org) (formally IPython) environment in their web browser. The current Amazon Web Services (AWS) documentation covers spinning up a Spark cluster in detail [here](https://docs.aws.amazon.com/ElasticMapReduce/latest/DeveloperGuide/emr-spark-launch.html) but not the Jupyter setup which allows you to run pySpark in your web browser. This guide shows you how to setup pySpark in Jupyter on an EMR cluster with Spark installed via a bootstrap action or as an EMR Application from a Windows machine, using only the [AWS CLI](http://docs.aws.amazon.com/cli/latest/reference/). Of course the AWS console can be used but takes longer as it involves using the Web user interface and makes automation more complex!

## Use of Spark in the JustGiving RAVEN platform
![JustGiving RAVEN](raven.png "JustGiving RAVEN")

Apache Spark is a fast and computing engine for large-scale data processing supported on [AWS EMR](http://aws.amazon.com/elasticmapreduce/). RAVEN is an adaptable AWS cloud computing event-driven platform that supports evolving reporting and adhoc analytic requirements at JustGiving. In our data science team we have been using Spark on EMR. EMR is a managed [Apache Hadoop](https://hadoop.apache.org/) framework that has allowed us to run large automated ETL and analytics processes on terabytes of data quickly and without worrying about the infrastructure setup and maintenance behind the cluster. Before EMR our data scientists were limited to algorithms that run on a single machine and were only able to work on sample data sets, anything larger required days of computation. Now with EMR running Hadoop or Spark clusters, our data scientists can easily launch clusters with 100s of EC2 machines to compute scalable text analytics and machine learning algorithms. For example we use natural language processing, on large volumes of unstructured data to automatically annotate charities.

## Setup AWS and PuTTY on Windows
1. Install and Setup the AWS CLI
 * Install the 64-bit version of the [AWS CLI](http://aws.amazon.com/cli/)
 * run **aws-configure** and add the AWS public and private keys by running
 * press enter for the region and output format (optional)
```
aws configure
```
See [CLI docs](http://docs.aws.amazon.com/cli/latest/userguide/cli-chap-getting-started.html) for details
2. Install and setup Putty
 * Download [PuTTY](http://www.chiark.greenend.org.uk/~sgtatham/putty/download.html) to your computer

## Bucket and file setup
1. Create a new bucket, e.g. myjgbucket
2. If you have CR characters in the install-jupyter-notebook-pySpark.sh bootstrap file this can lead to issues in LINUX. Replace any CRLF character with LF, e.g. you can use notepad++ and replace extended "\r" with blanks
2. Copy the folders **data** and **bootstrap** to the bucket, you can AWS CLI or [S3 Browser](http://s3browser.com/download.php)


## Create Cluster from AWS CLI
There are two ways to launch a Spark cluster with Jupyter

### 1. Installing Spark as an EMR application
Launch EMR Spark v1.3.1 cluster launch

`
aws.exe emr create-cluster --name Poc-Spark-jupyter-Cluster-3.8.0 --ami-version 3.8.0 --instance-type m3.xlarge --instance-count 3 --applications Name=GANGLIA Name=SPARK,Args=[-g,-d,spark.executor.memory=10g] --bootstrap-actions  Path=s3://myjgbucket/bootstrap/install-jupyter-notebook-pySpark.sh,Name=Install_jupyter_NB --region eu-west-1 --use-default-roles --ec2-attributes KeyName=myKey --tags Name=poc-spark-node Owner=Data-Science-Team --enable-debugging --log-uri s3://myjgbucket/Log/
`

There are a number of parameters that can be changed
 * --name Poc-Spark-jupyter-Cluster-3.8.0
 * --instance-type - the [type of instances](http://aws.amazon.com/elasticmapreduce/pricing/)
 * --instance-count - the number of instances or nodes in your spark cluster.

### 2. Installing Spark via a bootstrap action

Using the bootstrap allows you to select the Spark version, e.g Spark v1.4.0b

`
aws.exe emr create-cluster --name Poc-Spark-jupyter-Cluster-3.8.0 --ami-version 3.8.0 --instance-type m3.xlarge --instance-count 3 --applications Name=GANGLIA --bootstrap-actions  Path=s3://support.elasticmapreduce/spark/install-spark,Name=Spark,Args=-v1.4.0.b   Path=s3://myjgbucket/bootstrap/install-jupyter-notebook-pySpark.sh,Name=Install_jupyter_NB --region eu-west-1 --use-default-roles --ec2-attributes  KeyName=myKey --tags Name=poc-spark-node Owner=Data-Science-Team --enable-debugging --log-uri s3://myjgbucket/Log/
`

There are a number of parameters that can be changed
 * --name Poc-Spark-jupyter-Cluster-3.8.0
 * --instance-type - the [type of instances](http://aws.amazon.com/elasticmapreduce/pricing/)
 * --instance-count - the number of instances or nodes in your spark cluster.
 * Args=-v1.4.0.b" if you want another version of Spark

## Spark cluster management

The provisioning process takes about 5/10 min with the current configration with full logging and monitoring tools.

When State = Running/Waiting the cluster is ready. Get the DNS end point so that you can SSH into the headnode
```
aws emr describe-cluster --region eu-west-1 --cluster-id j-XXXXXXXXXXXXX
```

Remember to terminate the cluster when you are done, as you will incur costs for leaving it on!
Record the cluster-id as this can be used to TERMINATE the instance, for example assume that the cluster-id j-XXXXXXXXXXXXX:

```
aws emr terminate-clusters --region eu-west-1 --cluster-ids j-XXXXXXXXXXXXX
```

If you forget the cluster-id then you can run the following
```
aws emr --region eu-west-1 list-clusters --active
```

To work out the costs of running your cluster you can use the AWS calculator:
 * [Monthly calculator](http://calculator.s3.amazonaws.com/index.html) Select EMR and the number of instances and the type.

## SSH into the EMR head node (Windows)
 * Start PuTTY.
 * In the Category list, click Session.
    * In the Host Name field, type Add the DNS end point (e.g. hadoop@ec2-52-17-30-49.eu-west-1.compute.amazonaws.com)
	* Select SSH and Port 22
 * In the Category list, expand Connection > SSH, and then click Auth.
    * For Private key file for authentication, click Browse and select the private key file (poc-raven-emr.ppk) used to launch the cluster.
 * In the Connection set
    * Seconds between keepalive to 60
    * Check Disable Nagle and Enable TCP keepalive
 * In the Category list, expand Connection > SSH, and then click Tunnels.
    * In the Source port field, type 8157 (an unused local port).
    * Leave the Destination field blank.
    * Select the Dynamic and Auto options.
    * Click Add
 * Go back to session
	* Enter a name under Saved Session and press Save
	* Next time you logon to a cluster you can use this configuration and just need to change the Host Name
 * Click Open.
 * Click Yes to dismiss the security alert.

A full guide can be found here:
 * [emr-ssh-tunnel](https://docs.aws.amazon.com/ElasticMapReduce/latest/DeveloperGuide/emr-ssh-tunnel.html)
 * [emr-connect-master-node-ssh](https://docs.aws.amazon.com/ElasticMapReduce/latest/DeveloperGuide/emr-connect-master-node-ssh.html)

## Accessing the cluster web pages:

Connect to the Master node and setup a secure tunnel, for windows use PuTTY and FoxyProxy:

http://docs.aws.amazon.com/ElasticMapReduce/latest/DeveloperGuide/emr-connect-master-node-proxy.html

|Name of Interface	| URI |
| ------------- |:-------------:|
|Hadoop ResourceManager		|http://master-public-dns-name:9026/ |
|Hadoop HDFS NameNode		|http://master-public-dns-name:9101/ |
|Ganglia Metrics Reports	|http://master-public-dns-name/ganglia/|
|Jupyter 			|http://master-public-dns-name:8192|
|Spark Jobs |http://master-public-dns-name:4040|


## Running code on Spark Cluster
Access the Jupyter wiht URL:
 * Jupyter http://master-public-dns-name:8192
 * For a new Notebook press New > Python2


## Running pySpark interactivly
Wordcount on local file
```
import os

spark_home = os.environ.get('SPARK_HOME', None)
text_file = sc.textFile("file:///home/hadoop/spark/README.md")

word_counts = text_file \
    .flatMap(lambda line: line.split()) \
    .map(lambda word: (word, 1)) \
    .reduceByKey(lambda a, b: a + b)

word_counts.collect()

```
Wordcount on a file in S3
```
file = sc.textFile("s3://myjgbucket/data/raven-wikipedia-extract.txt")
print 'First line: ' + str(file.first())
print 'Total number of lines: ' + str(file.count())
lines = file.filter(lambda line: "raven" in line)
print 'Total number of lines with term "raven": ' + str(lines.count())
```

Using Machine Learning Library (MLlib) *k*-means algorithm
```
import sys
import numpy as np
from pyspark import SparkContext
from pyspark.mllib.clustering import KMeans

def parseVector(line):
    return np.array([float(x) for x in line.split(' ')])

# sc._jsc.hadoopConfiguration().set("fs.s3a.endpoint", "s3-eu-west-1.amazonaws.com")

data_path = "s3://raven-data/kmeans_data.txt"
k = int(5)
lines = sc.textFile(data_path)
data = lines.map(parseVector)
model = KMeans.train(data, k)
print "Final centers: " + str(model.clusterCenters)

```
## Jupyter with mathplotlib
```
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt

# evenly sampled time at 200ms intervals
t = np.arange(0., 5., 0.2)

# red dashes, blue squares and green triangles
plt.plot(t, t, 'r--', t, t**2, 'bs', t, t**3, 'g^')
plt.show()
```

## Running shell command

You can run any shell command using a prefix exclamation mark !

### Running python files from Jupyter

Running *k*-means example  
`!~/spark/bin/spark-submit --driver-memory 1G --master yarn-client --num-executors 3  ~/spark/examples/src/main/python/mllib/kmeans.py s3://myjgbucket/data/kmeans_data.txt 5`

You will see the pySpark logging and get the final output like:
`Final centers: [array([ 9.1,  9.1,  9.1]), array([ 0.05,  0.05,  0.05]), array([10.1  ,  10.9  ,   0.205]), array([ 10.2 ,  10.4 ,   0.22]), array([ 0.2, 0.2, 0.2])]`


Running wordcount an a file in S3  
`
!~/spark/bin/spark-submit --master yarn-client --num-executors 2 --driver-memory 1G ~/spark/examples/src/main/python/wordcount.py s3://myjgbucket/data/raven-wikipedia-extract.txt
`



## Importing new packages and apps
If you need to import packages or run applications, you can use run python and bash commands directly in Jupyter.

For example install PostgreSQL
```
!sudo yum install postgresql
```
Or to install a package
```
!pip install jinja
```

## To add new packages to the cluster:
Use addPyFile(path) to add a .py or .zip dependency for all tasks to be executed on this SparkContext in the future. The path passed can be either a local file, a file in HDFS (or other Hadoop-supported filesystems), or an HTTP, HTTPS or FTP URI. More details here [pySpark docs](http://spark.apache.org/docs/latest/api/python/pyspark.html)

## Uploading / Downloading files and data from the EMR cluster or head node
* EMR cluster - you can use a thick client called [WinSCP](http://winscp.net/eng/index.php) with the key mykey.ppk to access files on the headnode.
* S3 - Use [S3 Browser](http://s3browser.com/download.php)

## Credits
* Please contact Richard Freeman by email at JustGiving.com for any feedback or comments. See [Richard Freeman's website](http://www.rfreeman.net) for his blog and research articles.
* A thanks to Amo Abeyaratne at AWS for his support.

## License
Copyright (c) 2015 Giving.com Ltd, trading as JustGiving, or its affiliates. All Rights Reserved.
Licensed under the Apache License, Version 2.0 license. See LICENSE file in the project root for full license information.
