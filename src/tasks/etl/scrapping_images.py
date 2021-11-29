# %% Packages

import os
import pandas as pd
import datetime
import time
import random
from pyhocon import ConfigTree
from tqdm import tqdm
import re
import pycountry_convert as pc
import OpenSSL
from gazpacho import Soup
import urllib.request
from urllib.error import URLError
from typing import List, Tuple, Union, Dict
from selenium.webdriver.chrome.webdriver import WebDriver
from selenium.webdriver.remote.webelement import WebElement
import undetected_chromedriver.v2 as uc
from random_user_agent.user_agent import UserAgent
from random_user_agent.params import SoftwareName, OperatingSystem
from selenium.common.exceptions import NoSuchElementException

from src.base_classes.task import Task
from src.utils.logging import get_logger

# %% Logger

logger = get_logger()

# %% Code


class TaskScrappingImages(Task):

    name = "task_scrapping_images"
    dependencies = []

    def __init__(self, config: ConfigTree, re_scrape_data: bool) -> None:
        super().__init__(config, self.name)
        self.re_scrape_data = re_scrape_data

    def run(self):

        logger.info("Creating the driver")
        driver = self.create_driver()

        raw_meta_info_path = self.path_output.get_string("raw_meta_information")
        if self.re_scrape_data:
            logger.info("Expanding the website in order to scrape everything")
            expanded_driver = self.expand_website(driver)

            logger.info("Scrape the meta information and image info of every athlete")
            meta_information = self.scrape_information(expanded_driver)

            logger.info("Save the raw meta information")
            self.save_pickle(saving_path=raw_meta_info_path, file=meta_information)

        else:
            logger.info("Load meta information")
            meta_information = self.load_pickle(loading_path=raw_meta_info_path)

        logger.info("Enrich meta information")
        enriched_meta_information = self.process_meta_info(meta_information)

        logger.info("Save processed meta information")
        enrich_meta_info_path = self.path_output.get_string(
            "processed_meta_information"
        )
        self.save_pickle(
            saving_path=enrich_meta_info_path, file=enriched_meta_information
        )

    def create_driver(self) -> WebDriver:
        """This method loads the chrome driver, with which we are
        scraping the web-information.

        :return: Driver with which we scrape the information
        :rtype: WebDriver
        """

        # Generate random user agent
        user_agent = self.create_random_user_agent()

        # Setting up the driver
        executable_path = self.parameters.get_string("chrome_driver_path")
        options = uc.ChromeOptions()
        options.add_argument("--no-sandbox")
        options.add_argument("--headless")
        options.add_argument("--remote-debugging-port=9222")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument(f"user_agent={user_agent}")
        driver = uc.Chrome(executable_path=executable_path, options=options)

        # Stating the wanted url
        url = self.parameters.get_string("url")
        driver.get(url=url)
        assert driver.title is not "", "The driver did not load correctly"

        return driver

    def create_random_user_agent(self) -> UserAgent:
        """This method generates a random user agent, which will be provided to
        our driver. Through that method we hide our scraping behind a legit looking
        user.

        :return: An UserAgent with random settings
        :rtype: UserAgent
        """
        limit = self.parameters.get_int("user_agent_limit")
        software_names = [SoftwareName.CHROME.value]
        operating_systems = [OperatingSystem.WINDOWS.value, OperatingSystem.LINUX.value]
        user_agent_rotator = UserAgent(
            software_names=software_names,
            operating_systems=operating_systems,
            limit=limit,
        )
        user_agent = user_agent_rotator.get_random_user_agent()
        return user_agent

    def random_waiting_time(self) -> float:
        """Since when scraping too fast the website is blocking you from continuing
        scraping, we wait after each trigger for a random amount of time between
        0 and a specified interger number of seconds

        :return: The number of seconds to wait
        :rtype: float
        """
        max_seconds_sleep = self.parameters.get_float("max_seconds_sleep")
        min_seconds_sleep = self.parameters.get_float("min_seconds_sleep")
        return random.uniform(min_seconds_sleep, max_seconds_sleep)

    def expand_website(self, driver: WebDriver) -> WebDriver:
        """This method expands the website in order to fully expand it before scraping
        the entire content at once

        :param driver: Driver pointing at the desired website
        :type driver: WebDriver
        :return: The button to expand the website
        :rtype: WebElement
        """

        expanded_driver = self.maximal_expanding(driver)
        loaded_and_expanded_driver = self.load_images(expanded_driver)
        return loaded_and_expanded_driver

    def maximal_expanding(self, driver: WebDriver) -> WebDriver:
        """This method is expanding the website. This is done by repeatedly pressing
        the 'Show More' button and scrolling to the button of the page in order to
        trigger the Table to expand. In order to not getting banned from the website
        a random waiting timer is included which forces the program to wait for a
        random amount of time.

        :param driver: Driver pointing at the desired website
        :type driver: WebDriver
        :return: Driver with expanded website
        :rtype: WebDriver
        """

        button, button_status = self.retrieve_show_more_button(driver)
        while bool(button_status):
            time.sleep(self.random_waiting_time())
            number_of_divs, _, total_divs = self.count_number_of_entries(driver)
            logger.info(f"So far we have found {number_of_divs} athlete dividers")
            if button_status == "Loading":
                driver.execute_script("arguments[0].scrollIntoView();", total_divs[-1])
            else:
                driver.execute_script("arguments[0].click();", button)
            button, button_status = self.retrieve_show_more_button(driver)

        logger.info(f"In total we found {number_of_divs} athlete dividers")
        return driver

    def load_images(self, driver: WebDriver) -> WebDriver:
        """Goes over the expanded driver and gives the webpage time to load the images.
        If this would not be done, most images would have not been loaded until then.

        :param driver: Expanded driver of the website
        :type driver: WebDriver
        :return: Expanded driver with loaded images
        :rtype: WebDriver
        """

        number_of_divs, number_of_images, total_divs = self.count_number_of_entries(
            driver
        )
        logger.info(f"We start with having {number_of_images} images of athletes")

        # Scroll to the top divider and rest
        num_logging = self.parameters.get_int("number_iterations_before_logging")
        driver.execute_script("arguments[0].scrollIntoView();", total_divs[0])
        time.sleep(self.parameters.get_int("min_seconds_sleep"))
        for num, div in tqdm(enumerate(total_divs)):
            time.sleep(self.parameters.get_float("image_loading_time_in_seconds"))
            driver.execute_script("arguments[0].scrollIntoView();", div)

            if not bool(num % num_logging):
                _, number_of_images, _ = self.count_number_of_entries(driver)
                logger.info(
                    f"So far we have found {number_of_images} images "
                    f"in {number_of_divs} divs",
                )

        image_to_divs_ratio = round(number_of_images / number_of_divs, 2) * 100
        logger.info(f"In the end we found images for {image_to_divs_ratio}% of divs")
        return driver

    def retrieve_show_more_button(self, driver: WebDriver) -> WebElement:
        """This method looks for the button on the website and returns the button

        :param driver: Driver pointing at the desired website
        :type driver: WebDriver
        :return: The button to expand the website
        :rtype: WebElement
        """
        button = driver.find_element_by_css_selector(".button.load-more-button")
        button_status = driver.find_element_by_xpath(
            "//div[contains(@class, 'athlete-table__show-more')]"
        ).get_attribute("innerText")
        return (button, button_status)

    def count_number_of_entries(
        self, driver: WebDriver
    ) -> Tuple[int, List[WebElement]]:
        """This method counts the number of athletes the website is currently showing.
        This method comes handy when trying to gauge whether the website actually
        expanded. Furthermore, since there is a problem with images not having enough
        time to load, we also track the information how many images have loaded

        :param driver: Driver pointing at the website with the athletes
        :type driver: WebDriver
        :return: Showing the number of results and giving a list containing the
        web-elements
        :rtype: Tuple[int, List[WebElement]]
        """
        element = "athlete-table__row athlete-table__row--link link-underline-trigger"
        athlete_divs = driver.find_elements_by_xpath(f"//tr[@class='{element}']")
        athlete_images = driver.find_elements_by_css_selector(
            ".object-fit-cover-picture__img"
        )

        return (len(athlete_divs), len(athlete_images), athlete_divs)

    def scrape_information(self, driver: WebDriver) -> List[List[Union[str, float]]]:
        """This method scrapes the meta information of every athlete. Particularly
        we are scraping the nationality, gender, age and kind of sports from the
        website.

        :param driver: The webdriver only linking to the website
        :type driver: WebDriver
        :return: A list of lists containing the information of each athlete which
        was scrapable
        :rtype: List[List[Union[str, float, int]]]
        """

        logger.info("Clean output directory")
        self.clean_directory(self.path_output)

        logger.info("Getting all parameters for the scrapping ready")
        _, _, athlete_list = self.count_number_of_entries(driver)
        meta_list = []
        num_logging = self.parameters.get_int("number_iterations_before_logging")
        image_folder = os.path.join(self.path_output, "images")

        logger.info("Starting the data extraction")
        for num, element in tqdm(enumerate(athlete_list)):
            image_path = os.path.join(image_folder, f"athlete_{num}.png")
            info_list = self.extract_athlete_information(element, num, image_path)
            if bool(info_list):
                meta_list.append(info_list)

            if not bool(num % num_logging):
                logger.info(
                    f"We have scraped the information of {len(meta_list)} people "
                    f"out of {len(athlete_list)} total athletes"
                )

        return meta_list

    def extract_athlete_information(
        self, element: WebElement, num: int, image_path: str
    ) -> List[Union[str, float]]:
        """This method extracts the gender, age, sport and country information
        of each athlete. Furthermore, the image of the athlete is also taken.
        Of course, the entire information is only taken if we find an image and if
        we find exactly one piece of information for the athletes attributes. Meaning
        that if an athlete is conducting two sports, we do not consider this athlete's
        information. Further, if there is no information available for ANY of the
        athelete's attribute or if there is no image, no data is collected at all.

        :param element: Webelement containing the information of one athlete divider
        :type element: WebElement
        :param num: Counting variable which helps for saving purposes
        :type num: int
        :param image_path: Path at which the image should be saved
        :type image_path: str
        :return: Athlete's attributes summarized in a list
        :rtype: List[Union[str, float]]
        """

        img_src = self.extract_image_src(element)
        if bool(img_src):

            try:
                country = self.extract_country_information(element)
                sport = self.extract_sport_information(element)
                gender, age = self.extract_gender_and_age_information(element)

                required_length = 1
                bool_same_length = [
                    len([ele]) == required_length
                    for ele in [country, sport, gender, age]
                ]

                if bool_same_length:
                    urllib.request.urlretrieve(img_src, image_path)
                    return [num, country, sport, gender, age]

            except (NoSuchElementException, URLError, OSError) as e:
                logger.info(
                    f"Scraping information for athlete number {num}"
                    f"failed because {e}"
                )

    def extract_country_information(self, webelement: WebElement) -> str:
        """This method extracts the country information of the webelement

        :param webelement: Webelement of a certain athlete
        :type webelement: WebElement
        :return: Indication of country
        :rtype: str
        """
        country = webelement.find_element_by_css_selector(
            ".athlete-table__country"
        ).text
        return country

    def extract_gender_and_age_information(
        self, webelement: WebElement
    ) -> Tuple[str, float]:
        """This method extracts the information of gender and age of the athlete

        :param webelement: Webelement of a certain athlete
        :type webelement: WebElement
        :return: Tuple containing the information about the gender and age
        :rtype: Tuple[str, float]
        """
        gender_and_age = webelement.find_elements_by_css_selector(
            ".athlete-table__cell.u-hide-tablet.u-text-center"
        )
        gender, birth_date = (i.get_attribute("innerText") for i in gender_and_age)
        age = (
            datetime.datetime.today()
            - datetime.datetime.strptime(birth_date, "%d/%m/%Y")
        ).days / 365.2425
        return (gender, age)

    def extract_sport_information(
        self, webelement: WebElement
    ) -> Union[str, List[str]]:
        """This method extracts the sport of the athlete

        :param webelement: Webelement of a certain athlete
        :type webelement: WebElement
        :return: String if athlete is doing one sport and list if athlete is doing
        multiple sports
        :rtype: Union[str, List[str, str]]
        """
        sport = webelement.find_element_by_css_selector(
            ".athlete-table__discipline"
        ).text
        return sport

    def extract_image_src(self, webelement: WebElement) -> str:
        """This method extracts the image of the athlete if possible. If that action
        was not possible, the method returns none

        :param webelement: Webelement containing the information about the athlete
        :type webelement: WebElement
        :return: String at which the enlarged image of the athlete is to be found
        :rtype: str
        """
        try:
            image_element = webelement.find_element_by_css_selector(
                ".object-fit-cover-picture__img"
            )
            img_src = image_element.get_attribute("src")
            return re.sub("width=80", "width=350", img_src)
        except NoSuchElementException:
            return None

    def process_meta_info(
        self, meta_information: List[List[Union[str, float]]]
    ) -> pd.DataFrame:
        """The raw meta information is in a list of lists format. This format is rather
        undesirable, since it is so unstructured. For that reason we save the object
        as a dataframe. We furthermore enrich this dataset then with the information
        about the continent the athlete is coming from.

        :param meta_information: [description]
        :type meta_information: List[List[Union[str, float]]]
        :return: Meta information in a dataframe format and with continent information
        :rtype: pd.DataFrame
        """

        logger.info("Turn meta information into a dataframe")
        columns = ["number", "country_code", "sports", "gender", "age"]
        meta_df = pd.DataFrame.from_records(data=meta_information, columns=columns)

        logger.info("Remove invalid country codes and corresponding image")
        meta_df_cleaned, list_invalid_numbers = self.remove_invalid_country_codes(
            meta_df
        )
        self.remove_invalid_country_images(list_invalid_numbers)

        logger.info("Retrieving and adding the continent information")
        country_code_list = meta_df_cleaned.loc[:, "country_code"].tolist()
        continent_list = self.load_continent_information(country_code_list)
        meta_df_cleaned.loc[:, "continent"] = continent_list

        return meta_df_cleaned

    def remove_invalid_country_codes(
        self, meta_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, List[int]]:
        """Some country codes are not coming from official countries, such as
        ROT - "Refugee Olympic Team". For that reason we have to remove the athletes
        which contains such information before proceeding.

        :param meta_df: Meta information in dataframe format
        :type meta_df: pd.DataFrame
        :return: DataFrame with valid country codes and list of invalid country codes
        :rtype: Tuple[pd.DataFrame, List[int]]
        """

        invalid_noc_codes = self.parameters.get_list("invalid_noc_codes")
        bool_invalid_countries = meta_df.loc[:, "country_code"].isin(invalid_noc_codes)

        meta_df_cleaned = meta_df.loc[~bool_invalid_countries, :]
        list_invalid_image_numbers = meta_df.loc[
            bool_invalid_countries, "number"
        ].tolist()

        return (meta_df_cleaned, list_invalid_image_numbers)

    def remove_invalid_country_images(self, list_invalid_numbers: List[int]) -> None:
        """Since some country codes had to be removed, since they are not proper
        countries, such as 'ROT'. We should therefore not forget to also remove
        the corresponding image of these athletes.

        :param list_invalid_numbers: List with number of invalid athletes
        :type list_invalid_numbers: List[int]
        """

        image_data_path = self.path_output.get_string("image_data")
        for invalid_number in list_invalid_numbers:
            file_name = f"athlete_{invalid_number}.png"
            invalid_image_path = os.path.join(image_data_path, file_name)
            if os.path.isfile(invalid_image_path):
                os.remove(invalid_image_path)

    def load_continent_information(self, country_code_list: List[str]) -> List[str]:
        """In order to classify the broader origin of the athlete, we are adding
        the information from which continent the athlete is coming from. Getting
        this information is rather cumbersome. That is because our starting point
        is the NOC country code, which is anything but like the more commonly used
        ISO country codes which map then to continents. Hence some more work is invovled
        in that step.

        :param country_code_list: NOC country codes which we would like to translate
        :type country_code_list: List[str]
        :return: List of translated continent names of the NOC country codes
        :rtype: List[str]
        """

        noc_to_iso_dict = self.create_noc_to_iso_dict()
        alpha2_codes = [noc_to_iso_dict[x] for x in country_code_list]
        return [self.translate_alpha2_to_continent(x) for x in alpha2_codes]

    def translate_alpha2_to_continent(self, alpha2: str) -> str:
        """We are translating the ISO alpha2 country code to the continent's name
        by using this method.

        :param alpha2: Alpha 2 country code
        :type alpha2: str
        :return: Continent name
        :rtype: str
        """
        additional_iso_continent_dict = self.parameters.get_config(
            "additional_iso_continent_dict"
        )

        try:
            continent_code = pc.country_alpha2_to_continent_code(alpha2)
        except KeyError as e:
            continent_code = additional_iso_continent_dict.get(alpha2)
            logger.info(f"Since {e} we manually add the continent {continent_code}")
        return pc.convert_continent_code_to_continent_name(continent_code)

    def create_noc_to_iso_dict(self) -> Dict:
        """As outlined in the parent method, we have to map the NOC first to ISO codes
        in order to match them then with the continent name. The translation of
        NOC to ISO could not be found in a .csv format, therefore we scraped a table
        from the web to make the translation. Afterwards, we changed country-codes
        with unusual values.

        :return: Dictionary mapping the NOC code to the alpha 2 country code
        :rtype: Dict
        """

        logger.info("Scraping NOC to ISO2 translation table")
        noc_to_iso_dict = self.scrape_noc_iso_table()

        logger.info("Adding additional country codes")
        add_noc_country_config = self.parameters.get_config("additional_noc_iso_dict")
        add_noc_country_dict = dict(add_noc_country_config)

        return {**noc_to_iso_dict, **add_noc_country_dict}

    def scrape_noc_iso_table(self) -> Dict:
        """This method scrapes the mapping of the NOC country code to the ISO Alpha 2
        country code. We are using the gazpacho package for that reason.

        :return: Mapping of NOC to ISO alpha 2 information.
        :rtype: Dict
        """

        def parse_tr(tr):
            ioc_name = tr.find("td")[4].text
            country_name = tr.find("td")[1].text
            return ioc_name, country_name

        url = "https://www.worlddata.info/countrycodes.php"
        soup = Soup.get(url)

        table = soup.find("table", {"class": "std100 hover"}, mode="first")
        trs = table.find("tr")[1:]
        return {parse_tr(tr)[0]: parse_tr(tr)[1] for tr in trs}

    def clean_directory(self, path: str, keep_subfolders: bool) -> None:
        """This method cleans all potentially written output. This method ensures
        that we do not find some unexpected content from the last time this class
        was run. We do not want to delete folders, but only files within all
        subdirectories of the stated path.

        :param path: Directory which should be checked and cleaned
        :type path: str
        :param keep_subfolders: Indicating whether to keep the subfolders or delete
        everything
        :type keep_subfolders: bool
        """

        if keep_subfolders:
            for root, _, files in os.walk(path):
                for file in files:
                    file_path = os.path.join(root, file)
                    os.remove(file_path)
        else:
            for file in [x for x in os.listdir(path) if not x.startswith(".")]:
                total_path = os.path.join(path, file)
                shutil.rmtree(total_path)
