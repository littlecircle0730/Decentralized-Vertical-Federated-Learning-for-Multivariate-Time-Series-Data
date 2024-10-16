
# Code

This code implements Decentralized Federated Learning for Multivariate Time Series Prediction, featuring a novel weighted aggregation method. The requirements outlined below are derived primarily from the project's README file.

## Requirements

Experiments were done with the following package versions for Python 3.8 in Conda env:
conda install scikit-learn
conda install six
conda install psutil
conda install pandas
conda install numpy==1.21.4 matplotlib==3.4.3

This code should execute correctly with updated versions of these packages.

Prerequisites
Python Installation

Ensure you have a working installation of Python (preferably version 3.8).

SUMO Installation

SUMO is an open-source traffic simulation package. You can install SUMO on various operating systems by following the official installation guide: https://sumo.dlr.de/docs/Installing/index.html.

NUMO Installation

NUMO is a scenario package designed for use with SUMO to facilitate traffic simulations.

Clone the NUMO repository:
git clone https://github.com/ToyotaInfoTech/numo.git

Setting Up the Environment

Virtual Environment (Recommended)
It is recommended to create and use a virtual environment to ensure consistent dependencies and avoid conflicts.

Configure SUMO_HOME
Make sure the SUMO_HOME environment variable is set correctly:
export SUMO_HOME="/path/to/sumo"

Running the NUMO Simulation
jupyter notebook numo_data_collection_revised.ipynb

How each feature (Step, VehicleID, Obs_VehicleID, Obs_Speed, Obs_Acceleration, Obs_Direction, Obs_Location_X, Obs_Location_Y, TLS_X, TLS_Y, TLS_State, minDur, maxDur, timeSwitch) is recorded
1. Step
   - Description: The current time step in the simulation.
   - Recording Method: The current simulation time is recorded using current_time = traci.simulation.getTime() after each simulation step.

2. VehicleID
   - Description: The unique ID of the vehicle being processed.
   - Recording Method: The vehicle_id parameter in the collect_data function represents the current vehicle's ID and is recorded as part of the data.

3. Obs_VehicleID
   - Description: The unique ID of the observed (neighboring) vehicle.
   - Recording Method: The other_veh_id is obtained by checking nearby vehicles within a detection range and is recorded.

4. Obs_Speed
   - Description: The speed of the observed vehicle.
   - Recording Method: The speed is retrieved using traci.vehicle.getSpeed(other_veh_id) and recorded.

5. Obs_Acceleration
   - Description: The acceleration of the observed vehicle.
   - Recording Method: The acceleration is retrieved using traci.vehicle.getAcceleration(other_veh_id) and recorded.

6. Obs_Direction
   - Description: The direction of the observed vehicle (e.g., North, South, East, West).
   - Recording Method: The vehicle's angle is retrieved using traci.vehicle.getAngle(other_veh_id) and converted to a direction (N, E, S, W) using the get_direction(angle) function.

7. Obs_Location_X
   - Description: The X coordinate of the observed vehicle's location.
   - Recording Method: The X coordinate is obtained from traci.vehicle.getPosition(other_veh_id) and recorded.

8. Obs_Location_Y
   - Description: The Y coordinate of the observed vehicle's location.
   - Recording Method: The Y coordinate is obtained from traci.vehicle.getPosition(other_veh_id) and recorded.

9. TLS_X
   - Description: The X coordinate of the traffic light's location.
   - Recording Method: The traffic light's position is retrieved using traci.junction.getPosition(tls_id), and the X coordinate is recorded.

10. TLS_Y
    - Description: The Y coordinate of the traffic light's location.
    - Recording Method: Similarly, the Y coordinate from traci.junction.getPosition(tls_id) is recorded.

11. TLS_State
    - Description: The current state of the traffic light (e.g., red, green, yellow).
    - Recording Method: The traffic light state is retrieved using traci.trafficlight.getRedYellowGreenState(tls_id) and categorized into left turn, straight, and right turn directions.

12. minDur
    - Description: The minimum duration of the current traffic light phase.
    - Recording Method: Calculated using the calculate_duration(tls_id) function for each lane and recorded by direction (left turn, straight, right turn).

13. maxDur
    - Description: The maximum duration of the current traffic light phase.
    - Recording Method: Similarly calculated using calculate_duration(tls_id) and recorded by direction.

14. timeSwitch
    - Description: The time at which the traffic light state changes.
    - Recording Method: Updated and recorded using the updateSwitchTime(prev_data, current_time) function when a state change is detected.



### Core
The `losses` and `networks` files are from the "Unsupervised Scalable Representation Learning for Multivariate Time Series".
 - `losses` folder: implements the triplet loss in the cases of a training set
   with all time series of the same length, and a training set with time series
   of unequal lengths;
 - `networks` folder: implements encoder and its building blocks (dilated
   convolutions, causal CNN);
 - `utils.py` file: common functions for three prediction methods.
 - `orig_Centralized.py` file: centralized training method.
 - `orig_Fed.py` file: general FL with FedAge aggregation.
 - `orig_WeightedLoss.py` file: the FL with weighted score method for DVFL.
 - `NUMO_load_data.py` file: functions for handling the data loading. 
 - `numo_data_collection_revised` file: the file that we use to execute SUMO with NUMO scenario and collect the features for traffic light prediction.
 - `hei_WeightedLoss.py` file: a random vehicle is selected as the aggregator within each community to perform local aggregation in the hierarchical structure for DVFL.
- `hie_WLoss_Alg.py` file: includes the Ng selection formula for choosing the optimal Ng to construct the hierarchical structure for DVFL.
- `hie_WLoss_Aggregator_Selection.py` file: the aggregator assignment algorithm is used to assign aggregators based on a max-heap built with randomly assigned CPU frequencies.

## Extra Files
 - `Attention*` files: All the files are further applied attention mechanism. These are not includes in the eventual result because the model become complex and takes much more time and rounds to train. However, it should be albe to reduce the MSE.

### Training on the UCR and UEA archives

For example, to train a causal dCNN mode with weighted loss:
`python ./WeightedLoss.py --log_filename ./Weighted.txt --processed_data_filename 'cent1' --num_total_clients 100 --learning_rate 0.003 --batch_size 16 --num_epoch 3 --isCommu True --isDCNN 1 --raw_data_filename 'cent1' --num_round 100`

Note:
1. There is no --num_round in Centralized
2. If no V2V communication, just remove `--isCommu True`. Do not put `--isCommu False`# DVFL
