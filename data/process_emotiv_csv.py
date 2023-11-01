import pandas as pd
import argparse
import os
import numpy as np

def process_data(experiment_name, trial):
    # Load the intervalMarker data
    print(f'raw/{experiment_name}/{trial}/eegData.csv');
    interval_data = pd.read_csv(f'raw/{experiment_name}/{trial}/eegMetadata.csv')

    # Load the neural data, skipping the first two metadata rows
    neural_data = pd.read_csv(f'raw/{experiment_name}/{trial}/eegData.csv', skiprows=1)

    # Extract timestamp and the values of interest
    print(neural_data.columns)
    neural_data_interest = neural_data.iloc[:, [0] + list(range(4, 18))]

    # Filter the interval_data for records of interest
    relevant_markers = ['recording', 'recording_eyes_closed']
    relevant_baseline_types = ['eyesopen_element', 'eyesclose_element']
    filtered_data = interval_data[(interval_data['marker_value'].isin(relevant_markers))]
    baseline_data = interval_data[
        interval_data['type'].apply(lambda x: any(x.startswith(prefix) for prefix in relevant_baseline_types))
    ]
    baseline_data = baseline_data.reset_index(drop=True)

    print(baseline_data.iterrows())
    output_dir = f'processed/csv_raw/{experiment_name}/{trial}/'
    for index, row in baseline_data.iterrows():
        eyesclosed_csv = f"{output_dir}baseline_eyesclosed.csv"
        eyesopen_csv = f"{output_dir}baseline_eyesopen.csv"

        start_time = row['timestamp']
        end_time = start_time + row['duration']
        relevant_neural_data = neural_data_interest[(neural_data_interest.iloc[:, 0] >= start_time) & (neural_data_interest.iloc[:, 0] <= end_time)]
        relevant_neural_data = relevant_neural_data.head(1024)

        if index == 1:
            with open(eyesopen_csv, 'w') as f:
                relevant_neural_data.to_csv(f, index=False, header=False)
        else:
            with open(eyesclosed_csv, 'w') as f:
                relevant_neural_data.to_csv(f, index=False, header=False)

    for index, row in filtered_data.iterrows():
        # Find the parent type for the current record
        parent_type_index = interval_data[
            (interval_data['timestamp'] < row['timestamp']) & 
            (interval_data['type'].str.startswith('phase_Stimuli'))
        ].iloc[-1]
        parent_type = parent_type_index['type']

        # Create the header for the CSV file
        header = {
            'type': parent_type,
            'marker_value': row['marker_value'],
            'duration': row['duration'],
            'timestamp': row['timestamp']
        }

        # Extract the relevant neural data
        start_time = row['timestamp']
        end_time = start_time + row['duration']
        relevant_neural_data = neural_data_interest[(neural_data_interest.iloc[:, 0] >= start_time) & (neural_data_interest.iloc[:, 0] <= end_time)]
        
        header_df = ['Timestamp', 'EEG.AF3', 'EEG.F7', 'EEG.F3', 'EEG.FC5', 'EEG.T7', 'EEG.P7', 'EEG.O1', 'EEG.O2', 'EEG.P8', 'EEG.T8', 'EEG.FC6', 'EEG.F4', 'EEG.F8', 'EEG.AF4']

        # Create the CSV file
        output_dir = f'processed/csv_raw/{experiment_name}/{trial}/'

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        filenameCsv = f"{output_dir}{parent_type}_{row['marker_value']}.csv"
        #filenameNpy = f"{output_dir}{parent_type}_{row['marker_value']}.npy"
        with open(filenameCsv, 'w') as f:
            f.write(','.join(header.keys()) + '\n')
            f.write(','.join(map(str, header.values())) + '\n')
            f.write(','.join(header_df) + '\n')
            relevant_neural_data.to_csv(f, index=False, header=False)


        #np.save(filenameNpy, relevant_neural_data.values.T)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process EEG data based on experiment name and trial.')
    parser.add_argument('experiment_name', type=str, help='Name of the experiment')
    parser.add_argument('trial', type=str, help='Trial name or number')
    
    args = parser.parse_args()
    
    process_data(args.experiment_name, args.trial)
