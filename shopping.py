import csv
import sys

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename):
  evidence = []
  labels = []
  
  month_mapping = {
          'Jan': 0,
          'Feb': 1,
          'Mar': 2,
          'Apr': 3,
          'May': 4,
          'Jun': 5,
          'Jul': 6,
          'Aug': 7,
          'Sep': 8,
          'Oct': 9,
          'Nov': 10,
          'Dec': 11
        }

  months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
  
  with open(filename, 'r') as file:
        reader = csv.reader(file)
        next(reader)
        
        for row in reader:
          administrative = int(row[0])
          administrative_duration = float(row[1])
          informational = int(row[2])
          informational_duration = float(row[3])
          product_related = int(row[4])
          product_related_duration = float(row[5])
          bounce_rates = float(row[6])
          exit_rates = float(row[7])
          page_values = float(row[8])
          special_day = float(row[9])
          month = [row[10]]
          operating_systems = int(row[11])
          browser = int(row[12])
          region = int(row[13])
          traffic_type = int(row[14])
          visitor_type = 1 if row[15] else 0
          weekend =  1 if row[16] else 0

        month = list(map(lambda month: month_mapping[month], months))
          
        evidence.append([
          administrative,
          administrative_duration,
          informational,
          informational_duration,
          product_related,
          product_related_duration,
          bounce_rates,
          exit_rates,
          page_values,
          special_day,
          month, 
          operating_systems,
          browser, 
          region, 
          traffic_type,
          visitor_type,
          weekend
        ])
          
        if row[17] == 'TRUE': 
          label = 1 
        else:
          label = 0
        labels.append(label)
        
            
  return (evidence, labels);

def train_model(evidence, labels):
    k=1
    neighboor = KNeighborsClassifier(n_neighbors=k)
    neighboor.fit(evidence, labels)
    return neighboor


def evaluate(labels, predictions):
  guess_positive = 0
  actual_positive = 0
  guess_negitive = 0
  actual_negitive = 0
  
  for index in range(len(labels)):
    if labels[index] == 1: actual_positive +=1 
    if predictions[index] == 1: guess_positive +=1
  
    if labels[index] == 0: actual_positive +=1 
    if predictions[index] == 0: guess_positive +=1
    
  return (guess_positive/actual_positive, guess_negitive/actual_negitive)


if __name__ == "__main__":
    main()
