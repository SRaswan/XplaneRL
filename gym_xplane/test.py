from xpc_older import XPlaneConnect
import gym_xplane.xpc_older as xp

clientAddr = "0.0.0.0"
xpHost='127.0.0.1'
xpPort=49009
clientPort=0
timeout=10000
max_episode_steps=300
client = xp.XPlaneConnect(clientAddr,xpHost,xpPort,clientPort,timeout ,max_episode_steps)
cnt = 0
while True:
    cnt = cnt + 1
    posi = client.getPOSI();
    ctrl = client.getCTRL();

    print("[%f]: Loc: (%4f, %4f, %4f) Aileron:%2f Elevator:%2f Rudder:%2f\n"\
       % (cnt, posi[0], posi[1], posi[2], ctrl[1], ctrl[0], ctrl[2]))
    if cnt >= 50:
        break

# pos = hi.getPOSI(0)
# print(pos)
