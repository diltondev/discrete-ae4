import re
import subprocess

# Change these values as needed
input_file = 'OD_matrix_2017.csv'
output_file = 'station_graph_w.dot'
REDUCE_ZONE_ONE = True
RENDER_STYLE = 2  # 0 for record, 1 for bidirectional edges, 2 for double edges

zone_one_stations = [
    "Aldgate",
    "Blackfriars",
    "Farringdon",
    "liverpool Street",
    "Old Street",
] if REDUCE_ZONE_ONE else [
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

if RENDER_STYLE == 0:
#For record shape with in on left and out on right:
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        outfile.write('digraph G {\nnode [shape=record];\n')
        nodeset = set()
        for line in infile:
            if regex.match(line):
                separated = line.strip().split(',')
                if separated[1] not in nodeset:
                    outfile.write(f'"{separated[1].strip()}"[label="{{<l> | {separated[1]} | <r>}}"];\n')
                    nodeset.add(separated[1])
                if separated[3] not in nodeset:
                    outfile.write(f'"{separated[3].strip()}"[label="{{<l> | {separated[3]} | <r>}}"];\n')
                    nodeset.add(separated[3])
                outfile.write(f'    "{separated[1].strip()}":r -> "{separated[3].strip()}":l [label="{separated[10]}",weight="{separated[10]}"];\n')
        outfile.write('rankdir=LR;\nranksep=2;\n}\n')
elif RENDER_STYLE == 1:
    # For bidirectional edges with wetights on both sides of edge
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        outfile.write('digraph G {\n')
        dirs = []
        for line in infile:
            if regex.match(line):
                separated = line.strip().split(',')
                dirs.append((separated[1].strip(), separated[3].strip(), separated[10]))
        while dirs.__len__() > 0:
           
            current = dirs.pop()
            opposite = next((d for d in dirs if d[0] == current[1] and d[1] == current[0]), None)
            if opposite:
                dirs.remove(opposite)
                outfile.write(f'    "{current[0].strip()}" -> "{current[1].strip()}" [dir="both",labeldistance=4,headlabel="{current[2]}",taillabel="{opposite[2]}",weight="{float(current[2]) + float(opposite[2])}"];\n')
            else:
                outfile.write(f'    "{current[0].strip()}" -> "{current[1].strip()}" [label="{current[2]}",weight="{current[2]}"];\n')
        # outfile.write('graph [sep="+0.4"];\n}\n')
        outfile.write('rankdir=LR;\nnodesep=0.6;\n}\n')
elif RENDER_STYLE == 2:
    # For regular directed edges, one for each direction
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        outfile.write('digraph G {\n')
        for line in infile:
            if regex.match(line):
                separated = line.strip().split(',')
                outfile.write(f'    "{separated[1].strip()}" -> "{separated[3].strip()}" [label="{separated[10]}",weight="{separated[10]}"];\n')
        outfile.write('\n}\n')
elif RENDER_STYLE == 3:
    # For writing filtered OD matrix to CSV
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        for line in infile:
            if regex.match(line):
                outfile.write(line)
else:
    print("Invalid RENDER_STYLE selected.")
if not RENDER_STYLE == 3: 
    subprocess.run(['dot', '-Tsvg', output_file, '-o', 'station_graph_w.svg'])