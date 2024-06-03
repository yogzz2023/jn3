import numpy as np
import math
import csv
import pandas as pd
from scipy.stats import chi2
import matplotlib.pyplot as plt

class CVFilter:
    def __init__(self):
        self.Sf = np.zeros((6, 1))  # Filter state vector
        self.Pf = np.eye(6)  # Filter state covariance matrix
        self.Sp = np.zeros((6, 1))
        self.plant_noise = 20  # Plant noise covariance
        self.H = np.eye(3, 6)  # Measurement matrix
        self.R = np.eye(3)  # Measurement noise covariance
        self.Meas_Time = 0  # Measured time
        self.Z = np.zeros((3, 1))

    def initialize_filter_state(self, x, y, z, vx, vy, vz, time):
        """Initialize filter state."""
        self.Sf = np.array([[x], [y], [z], [vx], [vy], [vz]])
        self.Meas_Time = time

    def initialize_measurement_for_filtering(self, x, y, z, mt):
        """Initialize measurement for filtering."""
        self.Z = np.array([[x], [y], [z]])
        self.Meas_Time = mt

    def predict_step(self, current_time):
        """Predict step of the Kalman filter."""
        dt = current_time - self.Meas_Time
        Phi = np.eye(6)
        Phi[0, 3] = dt
        Phi[1, 4] = dt
        Phi[2, 5] = dt
        Q = np.eye(6) * self.plant_noise
        self.Sp = np.dot(Phi, self.Sf)
        self.Pf = np.dot(np.dot(Phi, self.Pf), Phi.T) + Q

    def update_step(self):
        """Update step of the Kalman filter."""
        Inn = self.Z - np.dot(self.H, self.Sf)  # Calculate innovation directly
        S = np.dot(self.H, np.dot(self.Pf, self.H.T)) + self.R
        K = np.dot(np.dot(self.Pf, self.H.T), np.linalg.inv(S))
        self.Sf = self.Sf + np.dot(K, Inn)
        self.Pf = np.dot(np.eye(6) - np.dot(K, self.H), self.Pf)

def sph2cart(az, el, r):
    """Convert spherical coordinates to Cartesian coordinates."""
    x = r * np.cos(el * np.pi / 180) * np.sin(az * np.pi / 180)
    y = r * np.cos(el * np.pi / 180) * np.cos(az * np.pi / 180)
    z = r * np.sin(el * np.pi / 180)
    return x, y, z

def cart2sph(x, y, z):
    """Convert Cartesian coordinates to spherical coordinates."""
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    el = math.degrees(math.atan2(z, np.sqrt(x ** 2 + y ** 2)))
    az = math.degrees(math.atan2(y, x))

    if az < 0.0:
        az += 360

    return r, az, el

def cart2sph2(x, y, z, filtered_values_csv):
    """Convert multiple Cartesian coordinates to spherical coordinates."""
    r = []
    az = []
    el = []

    for i in range(len(filtered_values_csv)):
        r.append(np.sqrt(x[i]**2 + y[i]**2 + z[i]**2))
        el.append(math.atan(z[i] / np.sqrt(x[i]**2 + y[i]**2)) * 180 / math.pi)
        az_angle = math.atan(y[i] / x[i])

        if x[i] > 0.0:
            az_angle = math.pi / 2 - az_angle
        else:
            az_angle = 3 * math.pi / 2 - az_angle

        az_angle = az_angle * 180 / math.pi

        if az_angle < 0.0:
            az_angle = 360 + az_angle

        if az_angle > 360:
            az_angle = az_angle - 360

        az.append(az_angle)

    return r, az, el

def read_measurements_from_csv(file_path):
    """Read measurements from CSV file."""
    measurements = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header if exists
        for row in reader:
            mr = float(row[7])  # MR column
            ma = float(row[8])  # MA column
            me = float(row[9])  # ME column
            mt = float(row[10])  # MT column
            x, y, z = sph2cart(ma, me, mr)  # Convert spherical to Cartesian coordinates
            measurements.append((x, y, z, mt))
    return measurements

def group_measurements_into_tracks(measurements):
    """Group measurements into tracks."""
    tracks = []
    used_indices = set()
    for i, (x_base, y_base, z_base, mt_base) in enumerate(measurements):
        if i in used_indices:
            continue
        track = [(x_base, y_base, z_base, mt_base)]
        used_indices.add(i)
        for j, (x, y, z, mt) in enumerate(measurements):
            if j in used_indices:
                continue
            if abs(mt - mt_base) < 50:
                track.append((x, y, z, mt))
                used_indices.add(j)
        tracks.append(track)
    return tracks

def is_valid_hypothesis(hypothesis):
    """Check if a hypothesis is valid."""
    non_zero_hypothesis = [val for _, val in hypothesis if val != -1]
    return len(non_zero_hypothesis) == len(set(non_zero_hypothesis)) and len(non_zero_hypothesis) > 0

state_dim = 3  # 3D state (e.g., x, y, z)
chi2_threshold = chi2.ppf(0.95, df=state_dim)

def mahalanobis_distance(x, y, cov_inv):
    """Calculate Mahalanobis distance."""
    delta = y[:3] - x[:3]
    return np.sqrt(np.dot(np.dot(delta, cov_inv), delta))

def perform_clustering_hypothesis_association(tracks, reports, cov_inv):
    """Perform clustering, hypothesis generation, and association."""
    clusters = []
    for report in reports:
        distances = [np.linalg.norm(track - report) for track in tracks]
        min_distance_idx = np.argmin(distances)
        if distances[min_distance_idx] < chi2_threshold:
            clusters.append([min_distance_idx])
    print("Clusters:", clusters)

    hypotheses = []
    for cluster in clusters:
        num_tracks = len(cluster)
        base = len(reports) + 1
        for count in range(base ** num_tracks):
            hypothesis = []
            for track_idx in cluster:
                report_idx = (count // (base ** track_idx)) % base
                hypothesis.append((track_idx, report_idx - 1))
            if is_valid_hypothesis(hypothesis):
                hypotheses.append(hypothesis)

    probabilities = calculate_probabilities(hypotheses, tracks, reports, cov_inv)
    max_associations, max_probs = find_max_associations(hypotheses, probabilities, reports)

    for i, hypothesis in enumerate(hypotheses):
        print(f"Hypothesis {i+1}: {hypothesis}, Probability: {probabilities[i]}")

    for report_idx, association in enumerate(max_associations):
        if association != -1:
            print(f"Report {report_idx+1} associated with Track {association+1}, Probability: {max_probs[report_idx]}")

def calculate_probabilities(hypotheses, tracks, reports, cov_inv):
    """Calculate probabilities for each hypothesis."""
    probabilities = []
    for hypothesis in hypotheses:
        prob = 1.0
        for track_idx, report_idx in hypothesis:
            if report_idx != -1:
                distance = mahalanobis_distance(tracks[track_idx], reports[report_idx], cov_inv)
                prob *= np.exp(-0.5 * distance ** 2)
        probabilities.append(prob)
    probabilities = np.array(probabilities)
    probabilities /= probabilities.sum()
    return probabilities

def find_max_associations(hypotheses, probabilities, reports):
    """Find the most likely association for each report."""
    max_associations = [-1] * len(reports)
    max_probs = [0.0] * len(reports)
    for hypothesis, prob in zip(hypotheses, probabilities):
        for track_idx, report_idx in hypothesis:
            if report_idx != -1 and prob > max_probs[report_idx]:
                max_probs[report_idx] = prob
                max_associations[report_idx] = track_idx
    return max_associations, max_probs

def main():
    """Main processing loop."""
    kalman_filter = CVFilter()
    csv_file_path = 'ttk_84_2.csv'
    measurements = read_measurements_from_csv(csv_file_path)

    if not measurements:
        print("No measurements found in the CSV file.")
        return

    tracks = group_measurements_into_tracks(measurements)
    cov_inv = np.linalg.inv(np.eye(state_dim))  # Example covariance inverse matrix

    updated_states = []

    for group_idx, track_group in enumerate(tracks):
        print(f"Processing group {group_idx + 1}/{len(tracks)}")

        track_states = []
        reports = []

        for x, y, z, mt in track_group:
            if len(track_states) == 0:
                kalman_filter.initialize_filter_state(x, y, z, 0, 0, 0, mt)
            else:
                kalman_filter.initialize_measurement_for_filtering(x, y, z, mt)
                kalman_filter.predict_step(mt)
                kalman_filter.update_step()
            track_states.append(kalman_filter.Sf.copy())
            reports.append([x, y, z])

        updated_states.append(track_states)

    print(f"Updated States: {updated_states}")

    perform_clustering_hypothesis_association(tracks, reports, cov_inv)

    csv_file_predicted = "ttk_84.csv"
    df_predicted = pd.read_csv(csv_file_predicted)
    filtered_values_csv = df_predicted[['F_TIM', 'F_X', 'F_Y', 'F_Z']].values

    A = cart2sph2(filtered_values_csv[:,1], filtered_values_csv[:,2], filtered_values_csv[:,3], filtered_values_csv)
    number = 1000
    result = np.divide(A[0], number)
    print("Result:", result)

if __name__ == "__main__":
    main()
