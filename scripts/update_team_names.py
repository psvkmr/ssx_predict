import sys
from pathlib import Path

path = Path(__file__).parent.parent
ext = "/data/resources/team_name_changes.csv"
path = Path(str(path) + ext)

def add_team_name(ssx_name, xg_data_name):
    with open(path, 'a') as f:
        f.write(f'{ssx_name},{xg_data_name}\n')
        
add_team_name(sys.argv[1], sys.argv[2])
        
    