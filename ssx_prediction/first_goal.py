import config
from bs4 import BeautifulSoup
from statistics import mean

class first_goal:

    def __init__(self):
        self.driver = config.driver
    
    def get_fg_data(self, class_tag='fc-cta-consent'):
        self.fg_soups = []
        for fg_url in list(config.fg_urls.values()):
            self.driver.get(fg_url)
            fg_soup = BeautifulSoup(self.driver.page_source, 'html.parser')
            try:
                self.driver.find_element_by_class_name(class_tag).click()
            except:
                pass
            fg_soup = BeautifulSoup(self.driver.page_source, 'html.parser')
            self.fg_soups.append(fg_soup)

    def extract_fg_data(self):
        fg_dicts = []
        for fg_soup in self.fg_soups:
            fg_sub1 = fg_soup.findAll('table', attrs={'id': 'btable'})
            fg_sub2 = fg_sub1[7].find('tbody').findAll('tr', attrs={'class': 'odd'})
            fg_sub3 = [team_line.findAll('td') for team_line in fg_sub2]
            fg_sub4 = [[stat.get_text(strip=True) for stat in team_stat] for team_stat in fg_sub3]
            team_names = [team_stat[0] for team_stat in fg_sub4]
            team_ogs = [int(team_stat[2]) for team_stat in fg_sub4]
            team_dict = dict(zip(team_names, team_ogs))
            fg_dicts.append(team_dict)
        self.fg_dict = {**fg_dicts[0], **fg_dicts[1], **fg_dicts[2], **fg_dicts[3]}
        self.fg_dict.update({'Average': round(mean(list(self.fg_dict.values())))})

if __name__ == '__main__':
    fg_obj = first_goal()
    fg_obj.get_fg_data()
    fg_obj.extract_fg_data()
