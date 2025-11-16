import re

# Change these values as needed
input_file = 'OD_matrix_2017.csv'
output_file = 'station_graph.dot'


zone_one_stations = [
    "Aldgate",
    "Aldgate East",
    "Angel",
    "Baker Street",
    "Bank / Monument",
    "Barbican",
    "Battersea Power Station",
    "Bayswater",
    "Blackfriars",
    "Bond Street",
    "Borough",
    "Cannon Street",
    "Chancery Lane",
    "Charing Cross",
    "City Thameslink",
    "Covent Garden",
    "Earl's Court",
    "Edgware Road (Cir)",
    "Edgware Road (Bak)",
    "Elephant & Castle",
    "Elephant & Castle",
    "Embankment",
    "Euston",
    "Euston Square",
    "Farringdon",
    "Fenchurch Street",
    "Gloucester Road",
    "Goodge Street",
    "Great Portland Street",
    "Green Park",
    "High Street Kensington",
    "Holborn",
    "Hoxton",
    "Hyde Park Corner",
    "Kennington",
    "King's Cross St. Pancras",
    "Knightsbridge",
    "Lambeth North",
    "Lancaster Gate",
    "Leicester Square",
    "Liverpool Street",
    "London Bridge",
    "Mansion House",
    "Marble Arch",
    "Marylebone",
    "Moorgate",
    "Nine Elms",
    "Notting Hill Gate",
    "Old Street",
    "Oxford Circus",
    "Paddington",
    "Piccadilly Circus",
    "Pimlico",
    "Queensway",
    "Regent's Park",
    "Russell Square",
    "Shoreditch High Street",
    "Sloane Square",
    "South Kensington",
    "Southwark",
    "St. James's Park",
    "St. Pancras International",
    "St. Paul's",
    "Temple",
    "Tottenham Court Road",
    "Tower Gateway",
    "Tower Hill",
    "Vauxhall",
    "Victoria",
    "Warren Street",
    "Waterloo",
    "Westminster"
]


# Create a regex that matches any zone1 station as a separate CSV field

# Allow for varying spaces after station names
station_pattern = '|'.join(re.escape(n) + r'\s+' for n in zone_one_stations)
# Match second and fourth fields for station names, possibly separated by any value between
full_pattern = fr'^[^,]*,({station_pattern}),[^,]*,({station_pattern}),'

regex = re.compile(full_pattern, re.IGNORECASE)


regex = re.compile(full_pattern, re.IGNORECASE)
with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
    outfile.write('digraph G {\n')
    for line in infile:
        if regex.match(line):
            separated = line.strip().split(',')

            outfile.write(f'    "{separated[1]}" -> "{separated[3]}"[label="{separated[10]}",weight="{separated[10]}"];\n')
    outfile.write('}\n')
