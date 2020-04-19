import sys
import json

def main():
    if (len(sys.argv)) != 3:
        print("Usage: python3 main.py <lat> <long>")
        return

    lat = float(sys.argv[1])
    lng = float(sys.argv[2])

    data = json.dumps([
		{
			"latitude": "40.416775",
			"longitude": "-3.70379",
			"color": "GREEN",
			"weight": "6"
		},
		{
			"latitude": "41.385064",
			"longitude": "2.173403",
			"color": "GREEN",
			"weight": "2"
		},
		{
			"latitude": "52.130661",
			"longitude": "-3.783712",
			"color": "GREEN",
			"weight": "2"
		},
		{
			"latitude": "55.378051",
			"longitude": "-3.435973",
			"color": "GREEN",
			"weight": "8"
		},
		{
			"latitude": "-40.900557",
			"longitude": "-174.885971",
			"color": "GREEN",
			"weight": "6"
		},
		{
			"latitude": "40.714353",
			"longitude": "-74.005973",
			"color": "RED",
			"weight": "6"
		}
	])

    print(data)


    


if __name__ == "__main__":
    main()