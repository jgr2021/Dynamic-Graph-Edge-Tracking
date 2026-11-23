"""Extract the MILUV pose header timestamp and vertical position from one original ROS bag."""
import argparse
import csv
from pathlib import Path
from rosbags.highlevel import AnyReader

parser=argparse.ArgumentParser()
parser.add_argument('bag',type=Path)
parser.add_argument('robot',choices=['ifo001','ifo002','ifo003'])
parser.add_argument('csv',type=Path)
args=parser.parse_args()
topic=f'/{args.robot}/vrpn_client_node/{args.robot}/pose'
with AnyReader([args.bag]) as reader:
    connections=[c for c in reader.connections if c.topic==topic]
    if not connections:
        raise ValueError(f'Missing {topic} in {args.bag}')
    args.csv.parent.mkdir(parents=True,exist_ok=True)
    with args.csv.open('x',encoding='utf8',newline='') as stream:
        output=csv.writer(stream)
        output.writerow(['timestamp_ns','pz'])
        for c,bag_stamp,raw in reader.messages(connections=connections):
            msg=reader.deserialize(raw,c.msgtype)
            stamp=msg.header.stamp
            output.writerow([int(stamp.sec)*10**9+int(stamp.nanosec),msg.pose.position.z])
