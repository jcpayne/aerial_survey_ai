### Clone the AIDE repo and install libraries
```bash
# specify the root folder where you wish to install AIDE
targetDir=/home/sauron/aide

# create environment (requires conda or miniconda)
conda create -y -n aide python=3.7
conda activate aide

# download AIDE source code
#sudo apt-get update && sudo apt-get install -y git
cd $targetDir
#git clone -b multiProject https://github.com/microsoft/aerial_wildlife_detection.git #WRONG BRANCH
git clone https://github.com/microsoft/aerial_wildlife_detection.git

# install basic requirements
sudo apt-get update #ADDED (failed without it)
sudo apt-get install -y libpq-dev python-dev
cd aerial_wildlife_detection #ADDED (required)

#CHANGE REQUIRED: I edited these lines of requirements.txt:
kombu==4.6.11
celery[redis,auth,msgpack]==4.4.7

#Then
export PYTHONPATH=.
pip install -U -r requirements.txt
```
 
### Set filepath (temporarily and permanently)
```bash
export AIDE_CONFIG_PATH=/home/sauron/aide/aerial_wildlife_detection/config/settings.ini
echo "export AIDE_CONFIG_PATH=/home/sauron/aide/aerial_wildlife_detection/config/settings.ini" | tee ~/.profile
```
## Install PostgreSQL

```bash
# specify postgres version you wish to use (must be >= 9.5)
version=10

# install packages
sudo apt-get update && sudo apt-get install -y wget
echo "deb http://apt.postgresql.org/pub/repos/apt/ $(lsb_release -cs)-pgdg main" | sudo tee /etc/apt/sources.list.d/pgdg.list
wget --quiet -O - https://www.postgresql.org/media/keys/ACCC4CF8.asc | sudo apt-key add -
sudo apt-get update && sudo apt-get install -y postgresql-$version


# update the postgres configuration with the correct port
sudo sed -i "s/\s*port\s*=\s[0-9]*/port = $dbPort/g" /etc/postgresql/$version/main/postgresql.conf


# modify authentication
# NOTE: you might want to manually adapt these commands for increased security; the following makes postgres listen to all global connections
sudo sed -i "s/\s*#\s*listen_addresses\s=\s'localhost'/listen_addresses = '\*'/g" /etc/postgresql/$version/main/postgresql.conf
echo "host    all             all             0.0.0.0/0               md5" | sudo tee -a /etc/postgresql/$version/main/pg_hba.conf > /dev/null


# restart postgres and auto-launch it on boot
sudo service postgresql restart
sudo systemctl enable postgresql


# If AIDE is run on MS Azure: TCP connections are dropped after 4 minutes of inactivity
# (see https://docs.microsoft.com/en-us/azure/load-balancer/load-balancer-outbound-connections#idletimeout)
# This is fatal for our database connection system, which keeps connections open.
# To avoid idling/dead connections, we thus use Ubuntu's keepalive timer:
if ! sudo grep -q ^net.ipv4.tcp_keepalive_* /etc/sysctl.conf ; then
    echo "net.ipv4.tcp_keepalive_time = 60" | sudo tee -a "/etc/sysctl.conf" > /dev/null
    echo "net.ipv4.tcp_keepalive_intvl = 60" | sudo tee -a "/etc/sysctl.conf" > /dev/null
    echo "net.ipv4.tcp_keepalive_probes = 20" | sudo tee -a "/etc/sysctl.conf" > /dev/null
else
    sudo sed -i "s/^\s*net.ipv4.tcp_keepalive_time.*/net.ipv4.tcp_keepalive_time = 60 /g" /etc/sysctl.conf
    sudo sed -i "s/^\s*net.ipv4.tcp_keepalive_intvl.*/net.ipv4.tcp_keepalive_intvl = 60 /g" /etc/sysctl.conf
    sudo sed -i "s/^\s*net.ipv4.tcp_keepalive_probes.*/net.ipv4.tcp_keepalive_probes = 20 /g" /etc/sysctl.conf
fi
sudo sysctl -p
```

Apparently, the postgresql port wasn't found by the command on line 44, above.  I think you have to specify $dbPort first.
```
Error: invalid line 63 in /etc/postgresql/10/main/postgresql.conf: port =                               # (change requires restart)
```
I copied it manually from the settings.ini file to postgresql.conf.

    Also I had to provide `$dbUser`, `$dbName` and `$dbPassword` (it is not clear from the instructions whether AIDE is supposed to find them automatically if they are not specified.)
```bash
dbPort='17685'
dbName='ailabeltooldb'
dbUser='ailabeluser'
dbPassword='aiLabelUser'
```
## Set up the database
```bash
sudo -u postgres psql -c "CREATE USER $dbUser WITH PASSWORD '$dbPassword';"
sudo -u postgres psql -c "CREATE DATABASE $dbName WITH OWNER $dbUser CONNECTION LIMIT -1;"
sudo -u postgres psql -c "GRANT CONNECT ON DATABASE $dbName TO $dbUser;"
sudo -u postgres psql -d $dbName -c "CREATE EXTENSION IF NOT EXISTS \"uuid-ossp\";"

# NOTE: needs to be run after init
sudo -u postgres psql -d $dbName -c "GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO $dbUser;"
```
### CHANGES 
1. Added the command to set PYTHONPATH;
2. Edited the version numbers of celery and kombu.five to avoid an error (see lines 22-23 above)
3. I added the lines sys.path.insert below (but that can be removed)
```bash
export PYTHONPATH=. 
python setup/setupDB.py
```

## AI backend
pip install celery[librabbitmq,redis,auth,msgpack]

## Setup RabbitMQ
```bash
# define RabbitMQ access credentials. NOTE: replace defaults with your own values
    username=aide
    password='Qu@K6,gb<7'

    # install RabbitMQ server
    sudo apt-get update && sudo apt-get install -y rabbitmq-server

    # optional: if the port for RabbitMQ is anything else than 5672, execute the following line:
    port=5672   # replace with your port
    sudo sed -i "s/^\s*#\s*NODE_PORT\s*=.*/NODE_PORT=$port/g" /etc/rabbitmq/rabbitmq-env.conf

```
    #NOTE: There was a 'b' missing in the final 'rabbit' (it was spelled 'rabit')
```bash

    # start RabbitMQ server
    sudo systemctl enable rabbitmq-server.service
    sudo service rabbitmq-server start

    # install Celery (if not already done)
    #conda activate aide
    #pip install celery[librabbitmq,redis,auth,msgpack]

    # add the user we defined above
    sudo rabbitmqctl add_user $username $password

    # add new virtual host
    sudo rabbitmqctl add_vhost aide_vhost

    # set permissions
    sudo rabbitmqctl set_permissions -p aide_vhost $username ".*" ".*" ".*"

    # restart
    sudo service rabbitmq-server stop       # may take a minute; if the command hangs: sudo pkill -KILL -u rabbitmq
    sudo service rabbitmq-server start
```
## Install Redis
```bash
sudo apt-get update && sudo apt-get install redis-server

# make sure Redis stores its messages in an accessible folder (we're using /var/lib/redis/aide.rdb here)
sudo sed -i "s/^\s*dir\s*.*/dir \/var\/lib\/redis/g" /etc/redis/redis.conf
sudo sed -i "s/^\s*dbfilename\s*.*/dbfilename aide.rdb/g" /etc/redis/redis.conf

# also tell systemd
sudo mkdir -p /etc/systemd/system/redis.service.d
echo -e "[Service]\nReadWriteDirectories=-/var/lib/redis" | sudo tee -a /etc/systemd/system/redis.service.d/override.conf > /dev/null

sudo mkdir -p /var/lib/redis
sudo chown -R redis:redis /var/lib/redis

# disable persistence. In general, we don't need Redis to save snapshots as it is only used as a result
# (and therefore message) backend.
sudo sed -i "s/^\s*save/# save /g" /etc/redis/redis.conf

# optional: if the port is anything else than 6379, execute the following line:
#   # replace with your port
#sudo sed -i "s/^\s*port\s*.*/port $port/g" /etc/redis/redis.conf

# restart
sudo systemctl daemon-reload
sudo systemctl enable redis-server.service
sudo systemctl restart redis-server.service
```

### Test Redis From AIController
```bash
  port=6379
  redis-cli -h localhost -p $port ping
  # > PONG
  redis-cli -h localhost -p $port set test "Hello, world"
  # > OK
```
### Test Redis from AIWorker
```bash
# replace the host and port accordingly
host=localhost          #     aicontroller.mydomain.net
port=6379

redis-cli -h $host -p $port ping
# > PONG
redis-cli -h $host -p $port get test
# > "Hello, world"
```
## Added lines to settings.ini
```bash
broker_URL = amqp://aide:'Qu@K6,gb<7'@localhost:5672/aide_vhost
result_backend = redis://localhost:6379/0
```

# Launching
Note: the documentation recommends adding these two lines to `~/.profile`, but the second line caused an error:  

**This failed**:
```bash
echo "export AIDE_CONFIG_PATH=config/settings.ini" | tee ~/.profile
echo "export AIDE_MODULES=LabelUI,AIController,FileServer,AIWorker" | tee ~/.profile
```

So I ended up doing this:

```bash
cd ~/aide/aerial_wildlife_detection
conda activate aide
export AIDE_CONFIG_PATH=config/settings.ini
export AIDE_MODULES=LabelUI
export PYTHONPATH=.     # might be required depending on your Python setup

./AIDE.sh start
```

Database params (copied here for convenience):
```bash
name = ailabeltooldb
schema = aerialelephants
host = localhost
port = 17685
user = 'ailabeluser'
password = 'aiLabelUser'
```


