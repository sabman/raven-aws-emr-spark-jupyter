# aws emr create-default-roles
# add roles to user http://docs.aws.amazon.com/ElasticMapReduce/latest/ManagementGuide/emr-iam-roles-creatingroles.html
aws emr create-cluster \
--name Spark-jupyter-Cluster-emr-4.7.0 \
--release-label emr-4.7.0 \
--instance-type m1.small \
--instance-count 1 \
--applications Name=Spark Name=GANGLIA \
--bootstrap-actions Path=s3://sabman-european-cities-analysis/install-jupyter-notebook-pySpark.sh,Name=Install_jupyter_NB \
--region eu-west-1 \
--use-default-roles \
--ec2-attributes KeyName=eu-west-1-spark \
--tags Name=spark-node Owner=Data-Science-Team \
--enable-debugging \
--log-uri s3://sabman-european-cities-analysis
