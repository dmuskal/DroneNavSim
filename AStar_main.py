from DroneClient import DroneClient
import time
import airsim.utils
import numpy as np
import math
import csv
import matplotlib.pyplot as plt


def rx(phi):
    return np.matrix([[1, 0, 0],
                      [0, math.cos(phi), math.sin(phi)],
                      [0, -math.sin(phi), math.cos(phi)]])


def ry(theta):
    return np.matrix([[math.cos(theta), 0, -math.sin(theta)],
                      [0, 1, 0],
                      [math.sin(theta), 0, math.cos(theta)]])


def rz(psi):
    return np.matrix([[math.cos(psi), math.sin(psi), 0],
                      [-math.sin(psi), math.cos(psi), 0],
                      [0, 0, 1]])


def my_b2i(phi, theta, psi):
    rot_x = rx(phi)
    rot_y = ry(theta)
    rot_z = rz(psi)
    i2b = rot_x * rot_y * rot_z
    b2i = np.transpose(i2b)
    return b2i


if __name__ == "__main__":
    client = DroneClient()
    client.connect()

    print(client.isConnected())

    # We now design the initial and final points and waiting until we are at the desired height
    # We assume no obstacle exists in 'pixel_len' radius around q_ini and q_final
    time.sleep(4)
    q_ini = (-400, -300, -50)
    q_fin = (-600, -800, -50)

    client.setAtPosition(q_ini[0], q_ini[1], q_ini[2])

    while True:
        # Waiting until achieving desired height sufficiently
        myPose = client.getPose()
        qz = myPose.pos.z_m
        d_from_ini = abs(qz - q_ini[2])
        if d_from_ini <= 1:
            print("Achieved desired height within threshold of 1 meter.\nBeginning to move toward destination.")
            break
        time.sleep(1)

    # We set the play field as grid to use the A-star
    field_len = 1500  # the squared area edge length im meters. total area is area_edge^2.
    x_boundary = (-1300, 200)
    y_boundary = (-1500, 0)
    pixel_len = 5  # grid the area to pixels with pixel_len
    pix_in_line = math.ceil(field_len / pixel_len)  # number of pixels in a line
    p_ini = (math.ceil((x_boundary[1] - q_ini[0]) / pixel_len), math.ceil((q_ini[1] - y_boundary[0]) / pixel_len))
    p_fin = (math.ceil((x_boundary[1] - q_fin[0]) / pixel_len), math.ceil((q_fin[1] - y_boundary[0]) / pixel_len))
    obs_map = np.zeros([pix_in_line, pix_in_line])
    route = np.array(p_ini)  # initialling the route with the first pixel
    lidar_time_step = 0.047  # time step in seconds to receive sensor measurement
    time_steps_ratio = 20  # Have to be an integer
    act_time_step = int(lidar_time_step * time_steps_ratio)  # time step in seconds to measure a new command
    time_step_counter = 0  # time step counter to help us recognize rather we should actuate new command
    v_f = 4  # velocity of free flying. Should be less then pixel_len
    prev_command = (0, 0, q_ini[2])  # Remembering the last command in order not to spam the quadcopter
    obs_added_flag = 1  # will be one if between two actuation time-step we observed new obstacle. Initialize with 1
    safety_distance = 1  # distance in meters to effectively enlarge the obstacles
    print("Initial pixel: " + str(p_ini))
    print("Goal pixel: " + str(p_fin))

    while True:
        # Advancing time steps
        time_step_counter = time_step_counter % time_steps_ratio + 1
        time.sleep(lidar_time_step)

        # Assessing our position and checking rather we finished
        myPose = client.getPose()
        q = (myPose.pos.x_m, myPose.pos.y_m, myPose.pos.z_m)
        p = (math.ceil((x_boundary[1] - q[0]) / pixel_len), math.ceil((q[1] - y_boundary[0]) / pixel_len))
        d = (np.linalg.norm(list(np.subtract(q, q_fin))))
        if d <= pixel_len:
            # We arrived to destination. We declare victory and exit
            print("Successfully achieved goal within " + str(pixel_len) + " meter radius.\nCurrent pose is "
                  + str(client.getPose()))
            plt.imshow(obs_map)
            break

        # If we arrived here, we need to advance forward.
        # First we check rather there is an obstacle in our environment.
        # We use the lidar and extract data much faster then the actuation time to avoid obstacles.

        xAngle = myPose.orientation.x_rad
        yAngle = myPose.orientation.y_rad
        zAngle = myPose.orientation.z_rad
        B2I = np.round(my_b2i(xAngle, yAngle, zAngle), decimals=4)  # Body to inertial rotation matrix
        lidar_data = client.getLidarData()  # distance from obstacles in body frame
        while len(lidar_data.points) > 1:
            obs_world_coo = np.matrix(q) + np.transpose(B2I * np.matrix([[lidar_data.points[0]],
                                                                         [lidar_data.points[1]],
                                                                         [lidar_data.points[2]]]))
            xy_unit = np.matrix([[lidar_data.points[0]], [lidar_data.points[1]], [0]]) / math.sqrt(
                pow(lidar_data.points[0], 2) + pow(lidar_data.points[1], 2))
            safety_addition = safety_distance * xy_unit
            safety_obs_world_coo = obs_world_coo + safety_addition
            pixel_obs_coo = (math.ceil((x_boundary[1] - safety_obs_world_coo.tolist()[0][0]) / pixel_len),
                             math.ceil((safety_obs_world_coo.tolist()[0][1] - y_boundary[0]) / pixel_len))
            if obs_map[pixel_obs_coo] < np.inf:
                obs_added_flag = 1  # indicating that we found new information
                obs_map[pixel_obs_coo] = np.inf  # setting the cost to be infinity so the algorithm won't go that way
            lidar_data.points = lidar_data.points[3:]

        # Now we check rather we also actuate in this time period
        if time_step_counter % time_steps_ratio != 0:
            continue

        print("Current position: " + str(q))
        print("Current pixel: " + str(p))

        # If we got here, we should perform an A* algorithm to determine our best step.
        closed_list = []  # list of all the pixels that already weighted
        opened_list = [p]  # list of all the pixels that should be considered as the next step
        g = np.array(obs_map)  # cost to move from start point to the pixel. We initiate to include the known obstacles.
        h = np.array(obs_map)  # cost to move from pixel to end point. We initiate to include the known obstacles.
        g[p] = 0  # In case we are near an obstacle we didn't see because of limited FOV
        h[p] = 0
        f = h + g
        finish_flag = 0  # is equal to 1 when the A* found the goal pixel
        finished_pix = (0, 0)  # is equal to the last curr_pix in the loop
        successors = {}  # indicating each pixel what is his successor - {successor: origin}
        while len(opened_list) > 0 and finish_flag == 0:
            open_dict = {opened_list[i]: f[opened_list[i]] for i in range(len(opened_list))}
            curr_pix = min(open_dict, key=open_dict.get)  # the current pixel is the one with the smallest f.
            cube_sub_f = [min(curr_pix[0], p_fin[0]), max((curr_pix[0], p_fin[0])),
                          min(curr_pix[1], p_fin[1]), max((curr_pix[1], p_fin[1]))]
            sub_f = f[cube_sub_f[0]:cube_sub_f[1], cube_sub_f[2]:cube_sub_f[3]]
            if sub_f.size:
                if np.max(sub_f) < np.inf and curr_pix != p:
                    # if we here, the box defined by the curr_pix and goal is empty, thus we finished
                    finish_flag = 1
                    finished_pix = curr_pix
                    break
            opened_list.remove(curr_pix)
            open_dict.pop(curr_pix)
            # for i in [-1, 0, 1]:
            # for j in [-1, 0, 1]:
            for m in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                # p_suc = (curr_pix[0] + i, curr_pix[1] + j)  # this is the successor of curr_pix
                p_suc = tuple(np.add(m, curr_pix))
                if curr_pix == p_suc:
                    continue
                if p_suc == p_fin:
                    finish_flag = 1
                    finished_pix = p_fin
                    successors.update({p_suc: curr_pix})
                    continue
                # if (curr_pix[0] + i == pix_in_line) or (curr_pix[1] + j == pix_in_line):
                if (curr_pix[0] + m[0] == pix_in_line) or (curr_pix[1] + m[1] == pix_in_line):
                    continue
                d_p2suc = pixel_len * (np.linalg.norm(list(np.subtract(curr_pix, p_suc))))
                d_suc2fin = pixel_len * (np.linalg.norm(list(np.subtract(p_suc, p_fin))))
                if g[p_suc] == np.inf:
                    d_p2suc = np.inf
                    d_suc2fin = np.inf
                temp_g = g[curr_pix] + d_p2suc
                temp_h = d_suc2fin
                temp_f = temp_g + temp_h
                if (p_suc in opened_list) and (f[p_suc] <= temp_f):
                    continue
                elif (p_suc in closed_list) and (f[p_suc] <= temp_f):
                    continue
                else:
                    g[p_suc] = temp_g
                    h[p_suc] = temp_h
                    f[p_suc] = temp_f
                    successors = {p_suc: curr_pix, **successors}
                    # successors.update({p_suc: curr_pix})
                    opened_list.append(p_suc)
                    if p_suc in closed_list:
                        closed_list.remove(p_suc)
            closed_list.append(curr_pix)

        # The algorithm has stopped and now we can find our next step
        if finish_flag == 0:
            print("No viable route could be calculated from current position to goal.")
            exit()

        origin_pix = finished_pix
        while True:
            if successors.get(origin_pix) == p:
                break
            else:
                origin_pix = successors.get(origin_pix)
        p_next = origin_pix
        q_next = (x_boundary[1] - (p_next[0] * pixel_len - pixel_len / 2),
                  (p_next[1] * pixel_len) + y_boundary[0] - pixel_len / 2, q_fin[2])

        # We only want a new command to quadcopter if the previous command was fulfilled or when we discovered a new
        # obstacle on the way
        via_q_dist = (np.linalg.norm(list(np.subtract(prev_command, q))))
        if obs_added_flag == 0 and via_q_dist >= pixel_len:
            time.sleep(act_time_step)
            continue

        #  If we got here we need to move to the center of p_next
        client.flyToPosition(q_next[0], q_next[1], q_next[2], v_f)
        print("Distance from goal: " + str(d))
        print("Now moving to next pixel: " + str(p_next))
        print("Now moving to next position: " + str((q_next[0], q_next[1], q_fin[2])))
        print("------------------------------------------------------------")
        prev_command = q_next
        obs_added_flag = 0  # Resetting the flag to know in the next actuation step rather we encountered new obstacle
        time.sleep(act_time_step)
