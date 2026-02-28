import numpy as np
from collections import deque

class SoccerAnalytics:
    def __init__(self):
        self.history = {}
        
        self.m_per_px = 0.04 

    def get_stats(self, p_id, curr_pos, fps=30):
        if p_id not in self.history:
            self.history[p_id] = {'coords': deque(maxlen=60), 'dist': 0, 'max_spd': 0}
        
        ph = self.history[p_id]
        ph['coords'].append(curr_pos)
        
        
        pace = 0
        if len(ph['coords']) > 1:
            px_dist = np.linalg.norm(curr_pos - ph['coords'][-2])
            meter_dist = px_dist * self.m_per_px
            ph['dist'] += meter_dist
            speed_kmh = meter_dist * fps * 3.6
            if speed_kmh > ph['max_spd']: ph['max_spd'] = speed_kmh
            pace = int(min(99, ph['max_spd'] * 2.8)) 

        
        phy = int(min(99, 60 + (ph['dist'] / 5)))

       
        return {
            "PAC": pace if pace > 0 else 60, 
            "SHO": 88 if pace > 25 else 75,
            "PAS": 85,
            "DRI": 90 if pace > 15 else 80,
            "DEF": 62,
            "PHY": phy
        }