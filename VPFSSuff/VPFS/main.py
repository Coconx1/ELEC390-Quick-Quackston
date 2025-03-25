import path_finding

def main():
    fare = path_finding.choose_fare()

    print("Selected Fare:", fare)
    path = path_finding.shortest_path(fare['src'], fare['dest'])

    print("Shortest Path:", path)

if __name__ == "__main__":
    main()
