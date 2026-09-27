from validation.pose_test import PoseTestRecorder

recorder = PoseTestRecorder(sample_count=5)
recorder.start_test("NEUTRAL")
for i in range(5):
    recorder.add_sample(0.1 * i, 0.2 * i, 0.3 * i)

print("Done")
