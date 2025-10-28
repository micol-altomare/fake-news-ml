import matplotlib.pyplot as plt
import time
import numpy as np
import requests
import os
import csv
from datetime import datetime
from tqdm import tqdm

"""
Goals:
1. Test classifier with 4 sentences
2. Test latency and create boxplot
"""

# Configuration
API_URL = "http://fakenewsml-env.eba-3zhcmbjj.us-east-2.elasticbeanstalk.com/"

# Test sentences
TEST_CASES = {
    "fake1": {
        "text": "Last month, Google acquired OpenAI for a whopping $600 billion.",
        "label": "FAKE"
    },
    "fake2": {
        "text": "Climate change is fake!!!!!",
        "label": "FAKE"
    },
    "real1": {
        "text": "Ocean acidification has significantly increased since the mid 1700s.",
        "label": "REAL"
    },
    "real2": {
        "text": "No prime minister in Canada has ever been impeached before.",
        "label": "REAL"
    }
}

NUM_ITERATIONS = 100
OUTPUT_DIR = "test_results"


def functional_test():
    """Run functional tests with the test cases."""
    print("\n" + "="*60)
    print("FUNCTIONAL TESTS")
    print("="*60)
    
    results = []
    
    for test_id, test_data in TEST_CASES.items():
        print(f"Input: {test_data['text']}")
        print(f"Expected: {test_data['label']}")
        
        try:
            response = requests.post(
                f"{API_URL}/predict", 
                json={"message": test_data["text"]},
                timeout=10
            )

            # FYI
            print(f"Response status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                predicted_label = result.get("label", "").upper()
                
                # Check if prediction matches expected
                passed = predicted_label == test_data["label"]
                status = "PASS" if passed else "FAIL"
                
                print(f"Predicted: {predicted_label}")
                print(f"Status: {status}")
                results.append(True)

            else:
                print(f"FAIL - Status code: {response.status_code}")
                print(f"Response: {response.text}")
                results.append(False)
        
        except Exception as e:
            print(f"FAIL - Error: {e}")
            results.append(False)

    # Summary
    print("\n" + "="*60)
    print("FUNCTIONAL TEST SUMMARY")
    print("="*60)
    print(f"Passed: {sum(results)}/{len(results)}")
    
    return results


def performance_test():
    """Run performance tests with 100 iterations per test case."""
    print("\n" + "="*60)
    print("PERFORMANCE TESTS")
    print("="*60)
    
    all_latencies = {}
    csv_files = []
    
    for test_id, test_data in TEST_CASES.items():
        print(f"\nRunning {NUM_ITERATIONS} iterations for: {test_id}")
        print(f"Message: {test_data['text']}")
        
        latencies = []
        csv_filename = os.path.join(OUTPUT_DIR, f"{test_id}_latencies.csv")
        csv_files.append(csv_filename)
        
        # Open CSV file for writing
        with open(csv_filename, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['iteration', 'timestamp', 'latency_ms', 'status_code', 'predicted_label'])
            
            for i in tqdm(range(NUM_ITERATIONS), desc=f"  {test_id}", unit="req"):
                try:
                    # Record start time
                    start_time = time.time()
                    timestamp = datetime.now().isoformat()
                    
                    # Make API call
                    response = requests.post(
                        f"{API_URL}/predict",
                        json={"message": test_data["text"]},
                        timeout=30)
                    
                    # Calculate latency
                    end_time = time.time()
                    latency_ms = (end_time - start_time) * 1000
                    
                    # Get prediction
                    if response.status_code == 200:
                        predicted_label = response.json().get("label", "ERROR")
                    else:
                        predicted_label = f"ERROR_{response.status_code}"
                    
                    latencies.append(latency_ms)
                    
                    # Write to CSV
                    csv_writer.writerow([
                        i + 1,
                        timestamp,
                        f"{latency_ms:.2f}",
                        response.status_code,
                        predicted_label
                    ])
                
                except Exception as e:
                    tqdm.write(f"  Error on iteration {i + 1}: {e}")
                    csv_writer.writerow([i + 1, timestamp, "ERROR", "N/A", "ERROR"])
        
        all_latencies[test_id] = latencies
        
        if latencies:
            avg_latency = np.mean(latencies)
            print(f"  Average latency: {avg_latency:.2f} ms")
            print(f"  CSV saved: {csv_filename}")


    return all_latencies


def generate_boxplots(all_latencies):
    """Generate boxplot of latency data."""
    
    data_to_plot = []
    labels = []
    
    for test_id in TEST_CASES.keys():
        if test_id in all_latencies and all_latencies[test_id]:
            data_to_plot.append(all_latencies[test_id])
            label = test_id.replace("_", " ").title()
            labels.append(label)
    
    # Create boxplot, display outliers
    fig, ax = plt.subplots(figsize=(8, 6))
    bp = ax.boxplot(data_to_plot, tick_labels=labels, showfliers=True)

    # Labels, title, grid
    ax.set_xlabel('Test Case')
    ax.set_ylabel('Latency (ms)')
    ax.set_ylim(50, 225)
    ax.set_title('API Latency (100 iterations per test case)')
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True) 
    plt.tight_layout() 
    
    # Save
    boxplot_filename = os.path.join(OUTPUT_DIR, "latency_boxplot.png")
    plt.savefig(boxplot_filename, dpi=300, bbox_inches='tight')
    print(f"Boxplot saved: {boxplot_filename}")
    
    return boxplot_filename


def main():
    # Create test output directory
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created output directory: {OUTPUT_DIR}")

    # Test if API is up
    try: 
        response = requests.get(f"{API_URL}/", timeout=10)
        if response.status_code == 200:
            print("Health check passed")
        else:
            print(f"Health check failed with status code: {response.status_code}")
    except Exception as e:
        print(f"Health check failed with error: {e}")
    
    # Run functional tests
    functional_test()
    
    # Run performance tests
    all_latencies = performance_test()
    
    # Generate boxplots
    generate_boxplots(all_latencies)


if __name__ == "__main__":
    main()