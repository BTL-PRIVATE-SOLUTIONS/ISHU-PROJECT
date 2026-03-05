"""Unified Dataset Loader for all meal planning datasets across all 5 data folders."""
import os
import pandas as pd
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
import difflib
import threading


class UnifiedDatasetLoader:
    """Load and manage all meal datasets from 5 data folders with comprehensive filtering."""
    
    def __init__(self, base_dir: str = None):
        """Initialize unified dataset loader for all data folders.
        
        Args:
            base_dir: Base directory containing all data folders (default: resolves to project/data)
        """
        # Resolve base_dir to absolute path relative to project root
        if base_dir is None:
            # Get the project root directory (parent of ai_engine directory)
            current_file = os.path.abspath(__file__)  # Path to this file
            ai_engine_dir = os.path.dirname(current_file)  # ai_engine directory
            project_root = os.path.dirname(ai_engine_dir)  # Parent of ai_engine (project root)
            base_dir = os.path.join(project_root, 'data')
        else:
            # If base_dir is relative, make it absolute relative to project root
            if not os.path.isabs(base_dir):
                current_file = os.path.abspath(__file__)
                ai_engine_dir = os.path.dirname(current_file)
                project_root = os.path.dirname(ai_engine_dir)
                base_dir = os.path.join(project_root, base_dir)
        
        self.base_dir = base_dir
        
        # Dataset configuration - maps folder paths to metadata
        self.dataset_configs = {
            'data_1': {
                'name': 'Regional Diets',
                'files': {
                    'northveg_cleaned.csv': {'region': 'North', 'diet': 'veg', 'category': 'regional'},
                    'northnonveg_cleaned.csv': {'region': 'North', 'diet': 'nonveg', 'category': 'regional'},
                    'northnonveg_cleaned (1).csv': {'region': 'North', 'diet': 'nonveg', 'category': 'regional'},
                    'southveg_cleaned.csv': {'region': 'South', 'diet': 'veg', 'category': 'regional'},
                    'southnonveg_cleaned.csv': {'region': 'South', 'diet': 'nonveg', 'category': 'regional'},
                }
            },
            'data_2': {
                'name': 'Trimester-Wise Diets',
                'files': {
                    'Trimester_Wise_Diet_Plan.csv': {'region': 'All', 'diet': 'all', 'category': 'trimester'},
                    'pregnancy_diet_1st_2nd_3rd_trimester.xlsx.csv': {'region': 'All', 'diet': 'all', 'category': 'trimester'},
                }
            },
            'data_3': {
                'name': 'Seasonal Diets',
                'files': {
                    'monsoon_diet_pregnant_women.csv': {'region': 'All', 'diet': 'all', 'season': 'monsoon', 'category': 'seasonal'},
                    'summer_pregnancy_diet.csv': {'region': 'All', 'diet': 'all', 'season': 'summer', 'category': 'seasonal'},
                    'Winter_Pregnancy_Diet.csv': {'region': 'All', 'diet': 'all', 'season': 'winter', 'category': 'seasonal'},
                }
            },
            'diabetiesdatasets': {
                'name': 'Diabetes-Pregnancy Specific Diets',
                'files': {
                    'diabetes_pregnancy_indian_foods.csv': {'region': 'All', 'diet': 'all', 'condition': 'diabetes', 'category': 'special_condition'},
                    'gestational_diabetes_indian_diet_dataset.csv': {'region': 'All', 'diet': 'all', 'condition': 'gestational_diabetes', 'category': 'special_condition'},
                    'Indian_Diabetes_Diet (1).csv': {'region': 'All', 'diet': 'all', 'condition': 'diabetes', 'category': 'special_condition'},
                }
            },
            'remainingdatasets': {
                'name': 'Specialized Dietary Guidance',
                'files': {
                    'foods_to_avoid_during_pregnancy_dataset.csv': {'region': 'All', 'diet': 'all', 'category': 'guidance', 'type': 'avoid_foods'},
                    'indian_diet_diabetes_pregnancy_dataset.csv': {'region': 'All', 'diet': 'all', 'condition': 'diabetes', 'category': 'special_condition'},
                    'postnatal_diet_india_dataset.csv': {'region': 'All', 'diet': 'all', 'phase': 'postnatal', 'category': 'postpartum'},
                    'postpartum_diet7_structured_dataset.csv': {'region': 'All', 'diet': 'all', 'phase': 'postpartum', 'category': 'postpartum'},
                    'pregnancy_diet_clean_dataset.csv': {'region': 'All', 'diet': 'all', 'category': 'general_pregnancy'},
                    'pregnancy_dos_donts_dataset.csv': {'region': 'All', 'diet': 'all', 'category': 'guidance', 'type': 'dos_donts'},
                }
            }
        }
        
        # Storage structures
        self.meals = []  # All meals loaded
        self.guidance = []  # Dos/Don'Ts and foods to avoid
        self.meals_by_category = defaultdict(list)  # Organize by category
        self.loading_stats = {'loaded': 0, 'failed': 0, 'by_folder': {}}
        
        # FAST LOOKUP CACHES - Initialize BEFORE loading (defensive initialization)
        self.food_index = {}  # Fast food name lookup
        self.keyword_index = defaultdict(list)  # Fast keyword-based lookup
        self.dos_donts_index = {}  # Fast dos/donts lookup
        self.lock = threading.Lock()  # Thread safety for caching
        
        # Meal preference cache for faster repeated queries
        self._preference_cache = {}
        self._preference_cache_max_size = 100  # Limit cache size
        
        # Load all datasets
        self._load_all_datasets()
        
        # Build fast lookup indexes
        try:
            self._build_fast_indexes()
            print(f"[OK] Built {len(self.food_index)} food indexes, {len(self.dos_donts_index)} guidance indexes")
        except Exception as e:
            print(f"[WARNING] Index building failed: {e}. Indexes available but may be empty.")
        
        self._print_loading_stats()
    
    def _load_all_datasets(self):
        """Load all datasets from all 5 data folders."""
        for folder_name, folder_config in self.dataset_configs.items():
            folder_path = os.path.join(self.base_dir, folder_name)
            self.loading_stats['by_folder'][folder_name] = {'loaded': 0, 'failed': 0}
            
            if not os.path.exists(folder_path):
                # Silently skip missing folders
                continue
            
            # Silently load data
            
            for filename, file_config in folder_config['files'].items():
                file_path = os.path.join(folder_path, filename)
                
                if os.path.exists(file_path):
                    try:
                        df = self._load_csv_file(file_path)
                        
                        if df is not None and len(df) > 0:
                            # Add metadata to rows
                            for col_name, col_value in file_config.items():
                                df[f'source_{col_name}'] = col_value
                            
                            # Categorize based on type
                            if file_config.get('type') in ['dos_donts', 'avoid_foods']:
                                self.guidance.extend(df.to_dict('records'))
                            else:
                                self.meals.extend(df.to_dict('records'))
                                self.meals_by_category[file_config['category']].extend(df.to_dict('records'))
                            
                            count = len(df)
                            self.loading_stats['loaded'] += count
                            self.loading_stats['by_folder'][folder_name]['loaded'] += count
                            # Suppress verbose output
                            pass
                        else:
                            self.loading_stats['by_folder'][folder_name]['failed'] += 1
                    except Exception as e:
                        self.loading_stats['by_folder'][folder_name]['failed'] += 1
                        # Suppress error output
                else:
                    self.loading_stats['by_folder'][folder_name]['failed'] += 1
                    # Suppress file not found messages
    
    def _load_csv_file(self, file_path: str) -> Optional[pd.DataFrame]:
        """Load CSV file with multiple encoding support."""
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1', 'ascii']
        
        for encoding in encodings:
            try:
                df = pd.read_csv(file_path, encoding=encoding, on_bad_lines='skip')
                
                # Clean column names
                df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]
                
                # Remove completely empty rows
                df = df.dropna(how='all')
                
                return df if len(df) > 0 else None
            except Exception:
                continue
        
        return None
    
    def _build_fast_indexes(self):
        """Build fast lookup indexes for instant query response."""
        print(f"[DEBUG] Building indexes from {len(self.meals)} meals and {len(self.guidance)} guidance items...")
        
        # Index all food items by name for instant lookup
        for meal in self.meals:
            # Extract potential food names from different column names
            for col_name in ['food', 'food_item', 'meal', 'dish', 'item', 'dish_name', 'meal_name', 'recipe']:
                if col_name in meal and meal[col_name]:
                    food_name = str(meal[col_name]).strip().lower()
                    if food_name:
                        self.food_index[food_name] = meal
                        # Also index by keywords
                        for word in food_name.split():
                            if len(word) > 2:
                                self.keyword_index[word].append(meal)
        
        # Index dos/donts for instant lookup
        for guidance in self.guidance:
            for col_name in ['item', 'food', 'food_item', 'do', 'dont']:
                if col_name in guidance and guidance[col_name]:
                    item_name = str(guidance[col_name]).strip().lower()
                    if item_name:
                        self.dos_donts_index[item_name] = guidance
        
        print(f"[DEBUG] Indexed {len(self.food_index)} foods, {len(self.keyword_index)} keywords, {len(self.dos_donts_index)} guidance items")
    
    def _print_loading_stats(self):
        """Print dataset loading statistics."""
        # Suppress verbose statistics output
        pass
    
    def search_food_exact(self, query: str) -> Optional[Dict]:
        """FAST: Search for exact food match in cached index (O(1) time).
        
        Args:
            query: Food name to search
            
        Returns:
            Food item if found, None otherwise
        """
        if not hasattr(self, 'food_index'):
            self.food_index = {}
        query_lower = query.strip().lower()
        return self.food_index.get(query_lower)
    
    def search_food_fuzzy(self, query: str, threshold: float = 0.7) -> List[Dict]:
        """FAST: Fuzzy search for foods matching query (cached).
        
        Args:
            query: Food name to search
            threshold: Similarity threshold (0.0-1.0)
            
        Returns:
            List of matching food items
        """
        if not hasattr(self, 'food_index'):
            self.food_index = {}
        query_lower = query.strip().lower()
        results = []
        checked = set()
        
        # First try exact match
        if query_lower in self.food_index:
            return [self.food_index[query_lower]]
        
        # Try fuzzy matching on cached food names
        for food_name in self.food_index.keys():
            if food_name not in checked:
                similarity = difflib.SequenceMatcher(None, query_lower, food_name).ratio()
                if similarity >= threshold:
                    results.append(self.food_index[food_name])
                    checked.add(food_name)
        
        return results[:10]  # Return top 10 matches
    
    def search_dos_donts_exact(self, query: str) -> Optional[Dict]:
        """FAST: Search for exact do's/don'ts match (O(1) time).
        
        Args:
            query: Item to search
            
        Returns:
            Do's/Don'ts item if found, None otherwise
        """
        if not hasattr(self, 'dos_donts_index'):
            self.dos_donts_index = {}
        query_lower = query.strip().lower()
        return self.dos_donts_index.get(query_lower)
    
    def search_dos_donts_fuzzy(self, query: str, threshold: float = 0.7) -> List[Dict]:
        """FAST: Fuzzy search for dos/donts matching query.
        """
        if not hasattr(self, 'dos_donts_index'):
            self.dos_donts_index = {}
        """
        
        Args:
            query: Item to search
            threshold: Similarity threshold
            
        Returns:
            List of matching items
        """
        query_lower = query.strip().lower()
        results = []
        checked = set()
        
        # First try exact match
        if query_lower in self.dos_donts_index:
            return [self.dos_donts_index[query_lower]]
        
        # Try fuzzy matching
        for item_name in self.dos_donts_index.keys():
            if item_name not in checked:
                similarity = difflib.SequenceMatcher(None, query_lower, item_name).ratio()
                if similarity >= threshold:
                    results.append(self.dos_donts_index[item_name])
                    checked.add(item_name)
        
        return results[:10]
    
    def quick_answer_from_cache(self, query: str) -> Dict:
        """SUPER FAST: Try to answer from cache in milliseconds.
        
        Returns dict with:
        - found: bool (True if answer found in cache)
        - data: dict (the cached data if found)
        - type: str ('food', 'dos_donts', or None)
        """
        # DEFENSIVE: Ensure indexes exist
        if not hasattr(self, 'food_index'):
            self.food_index = {}
        if not hasattr(self, 'dos_donts_index'):
            self.dos_donts_index = {}
        if not hasattr(self, 'keyword_index'):
            self.keyword_index = defaultdict(list)
        
        query_lower = query.strip().lower()
        
        # Try exact food match
        if query_lower in self.food_index:
            return {
                'found': True,
                'data': self.food_index[query_lower],
                'type': 'food'
            }
        
        # Try exact dos/donts match
        if query_lower in self.dos_donts_index:
            return {
                'found': True,
                'data': self.dos_donts_index[query_lower],
                'type': 'dos_donts'
            }
        
        # Try fuzzy match on foods
        fuzzy_foods = self.search_food_fuzzy(query, threshold=0.75)
        if fuzzy_foods:
            return {
                'found': True,
                'data': fuzzy_foods[0],
                'type': 'food'
            }
        
        # Try fuzzy match on dos/donts
        fuzzy_dos_donts = self.search_dos_donts_fuzzy(query, threshold=0.75)
        if fuzzy_dos_donts:
            return {
                'found': True,
                'data': fuzzy_dos_donts[0],
                'type': 'dos_donts'
            }
        
        # Not found - should use AI model
        return {
            'found': False,
            'data': None,
            'type': None
        }
    
    def get_meals_by_preference(self, 
                                region: Optional[str] = None,
                                diet_type: Optional[str] = None,
                                trimester: Optional[int] = None,
                                season: Optional[str] = None,
                                condition: Optional[str] = None,
                                meal_type: Optional[str] = None) -> List[Dict]:
        """Get meals matching specified preferences.
        
        Args:
            region: 'North' or 'South'
            diet_type: 'veg', 'nonveg', 'vegan'
            trimester: 1, 2, or 3
            season: 'summer', 'winter', 'monsoon'
            condition: 'diabetes', 'gestational_diabetes', etc.
            meal_type: 'breakfast', 'lunch', 'dinner', 'snack'
        
        Returns:
            List of matching meals
        """
        # Create cache key from preferences
        cache_key = f"{region}_{diet_type}_{trimester}_{season}_{condition}_{meal_type}"
        
        # Check cache first
        if cache_key in self._preference_cache:
            return self._preference_cache[cache_key]
        
        # If not in cache, compute results
        results = []
        normalized_region = self._normalize_region(region)
        normalized_diet = self._normalize_diet(diet_type)
        normalized_season = self._normalize_season(season)
        normalized_condition = self._normalize_condition(condition)
        normalized_meal_type = self._normalize_meal_type(meal_type)
        
        for meal in self.meals:
            match = True
            
            # Check regional preference
            # Include meals that match the region OR have "All" as region (generic meals)
            if normalized_region:
                source_region = meal.get('source_region')
                if source_region:
                    source_region_lower = source_region.lower()
                    # Include if: exact match OR source is "all" (generic for all regions)
                    if source_region_lower != normalized_region and source_region_lower != 'all':
                        match = False
            
            # Check diet type
            # Include meals that match the diet OR have "all" as diet (generic meals)
            if normalized_diet:
                source_diet = meal.get('source_diet')
                if source_diet:
                    source_diet_lower = source_diet.lower()
                    # Include if: exact match OR source is "all" (generic for all diets)
                    if source_diet_lower != normalized_diet and source_diet_lower != 'all':
                        match = False
            
            # Check meal type
            if normalized_meal_type:
                meal_col = self._find_meal_type_column(meal)
                if meal_col:
                    if meal.get(meal_col, '').lower() != normalized_meal_type:
                        match = False
            
            # Check special conditions
            # Include meals that match the condition OR don't have a specific condition
            if normalized_condition:
                source_condition = meal.get('source_condition')
                if source_condition:
                    if source_condition.lower() != normalized_condition:
                        match = False
            
            # Check season
            # Include meals that match the season OR have "all" as season (generic meals)
            if normalized_season:
                source_season = meal.get('source_season')
                if source_season:
                    source_season_lower = source_season.lower()
                    # Include if: exact match OR source is "all" (generic for all seasons)
                    if source_season_lower != normalized_season and source_season_lower != 'all':
                        match = False
            
            # Check trimester
            if trimester:
                trimester_col = self._find_trimester_column(meal)
                if trimester_col:
                    meal_trimester = meal.get(trimester_col)
                    if meal_trimester and str(trimester) not in str(meal_trimester):
                        match = False
            
            if match:
                results.append(meal)
        
        # Store in cache (limit cache size)
        if len(self._preference_cache) < self._preference_cache_max_size:
            self._preference_cache[cache_key] = results
        
        return results
    
    def _find_meal_type_column(self, meal: Dict) -> Optional[str]:
        """Find the column containing meal type information."""
        possible_columns = ['meal_type', 'type', 'meal', 'breakfast_lunch_dinner', 'meal_time']
        for col in possible_columns:
            if col in meal:
                return col
        return None
    
    def _find_trimester_column(self, meal: Dict) -> Optional[str]:
        """Find the column containing trimester information."""
        possible_columns = ['trimester', 'trimester_wise', 'month', 'week']
        for col in possible_columns:
            if col in meal:
                return col
        return None

    def _normalize_region(self, region: Optional[str]) -> Optional[str]:
        if not region:
            return None
        value = region.strip().lower()
        if 'north' in value:
            return 'north'
        if 'south' in value:
            return 'south'
        return value if value not in ['all', 'any'] else None

    def _normalize_diet(self, diet: Optional[str]) -> Optional[str]:
        if not diet:
            return None
        value = diet.strip().lower()
        if 'veg' in value and 'non' not in value:
            return 'veg'
        if 'non' in value:
            return 'nonveg'
        if 'vegan' in value:
            return 'vegan'
        return value if value not in ['mixed', 'all', 'any'] else None

    def _normalize_season(self, season: Optional[str]) -> Optional[str]:
        if not season:
            return None
        value = season.strip().lower()
        if 'summer' in value:
            return 'summer'
        if 'winter' in value:
            return 'winter'
        if 'monsoon' in value or 'rain' in value:
            return 'monsoon'
        return value if value not in ['all', 'any'] else None

    def _normalize_condition(self, condition: Optional[str]) -> Optional[str]:
        if not condition:
            return None
        value = condition.strip().lower()
        if 'gestational' in value:
            return 'gestational_diabetes'
        if 'diabetes' in value:
            return 'diabetes'
        return value if value not in ['general', 'none'] else None

    def _normalize_meal_type(self, meal_type: Optional[str]) -> Optional[str]:
        if not meal_type:
            return None
        value = meal_type.strip().lower()
        if value in ['snack', 'snacks']:
            return 'snack'
        return value
    
    def get_guidance(self, guidance_type: Optional[str] = None) -> List[Dict]:
        """Get guidance items (dos/donts, foods to avoid).
        
        Args:
            guidance_type: 'dos_donts' or 'avoid_foods', None for all
        
        Returns:
            List of guidance items
        """
        if guidance_type is None:
            return self.guidance
        
        return [g for g in self.guidance if g.get('source_type', '').lower() == guidance_type.lower()]
    
    def get_meals_for_condition(self, condition: str) -> List[Dict]:
        """Get meals suitable for specific health conditions."""
        return self.get_meals_by_preference(condition=condition)
    
    def get_seasonal_meals(self, season: str) -> List[Dict]:
        """Get meals recommended for a specific season."""
        return self.get_meals_by_preference(season=season)
    
    def get_trimester_meals(self, trimester: int) -> List[Dict]:
        """Get meals recommended for a specific trimester."""
        return self.get_meals_by_preference(trimester=trimester)
    
    def get_regional_meals(self, region: str) -> List[Dict]:
        """Get meals for a specific region."""
        return self.get_meals_by_preference(region=region)
    
    def get_statistics(self) -> Dict:
        """Get dataset statistics."""
        return {
            'total_meals': len(self.meals),
            'total_guidance': len(self.guidance),
            'categories': {cat: len(meals) for cat, meals in self.meals_by_category.items()},
            'by_region': self._count_by_field('source_region'),
            'by_diet': self._count_by_field('source_diet'),
            'by_condition': self._count_by_field('source_condition'),
            'by_season': self._count_by_field('source_season'),
        }
    
    def _count_by_field(self, field_name: str) -> Dict[str, int]:
        """Count meals by a specific field."""
        counts = defaultdict(int)
        for meal in self.meals:
            value = meal.get(field_name, 'unknown')
            if value:
                counts[value] += 1
        return dict(counts)
    
    def get_available_options(self) -> Dict[str, List[str]]:
        """Get all available options for user preference selection.
        
        Filters out generic values like 'All' and 'all' to show only specific options.
        """
        stats = self.get_statistics()
        return {
            # Only show specific regions (North, South), not 'All'
            'regions': sorted([r for r in stats['by_region'].keys() if r and r.lower() != 'all']),
            # Only show specific diets (veg, nonveg), not 'all'
            'diets': sorted([d for d in stats['by_diet'].keys() if d and d.lower() != 'all']),
            # Show all seasons
            'conditions': sorted([c for c in stats['by_condition'].keys() if c]),
            # Show all seasons
            'seasons': sorted([s for s in stats['by_season'].keys() if s]),
            'categories': list(self.meals_by_category.keys()),
        }

    """Load and manage all meal datasets from 5 data folders with comprehensive filtering."""
    
    def __init__(self, base_dir: str = None):
        """Initialize unified dataset loader for all data folders.
        
        Args:
            base_dir: Base directory containing all data folders (default: resolves to project/data)
        """
        # Resolve base_dir to absolute path relative to project root
        if base_dir is None:
            # Get the project root directory (parent of ai_engine directory)
            current_file = os.path.abspath(__file__)  # Path to this file
            ai_engine_dir = os.path.dirname(current_file)  # ai_engine directory
            project_root = os.path.dirname(ai_engine_dir)  # Parent of ai_engine (project root)
            base_dir = os.path.join(project_root, 'data')
        else:
            # If base_dir is relative, make it absolute relative to project root
            if not os.path.isabs(base_dir):
                current_file = os.path.abspath(__file__)
                ai_engine_dir = os.path.dirname(current_file)
                project_root = os.path.dirname(ai_engine_dir)
                base_dir = os.path.join(project_root, base_dir)
        
        self.base_dir = base_dir
        
        # Dataset configuration - maps folder paths to metadata
        self.dataset_configs = {
            'data_1': {
                'name': 'Regional Diets',
                'files': {
                    'northveg_cleaned.csv': {'region': 'North', 'diet': 'veg', 'category': 'regional'},
                    'northnonveg_cleaned.csv': {'region': 'North', 'diet': 'nonveg', 'category': 'regional'},
                    'northnonveg_cleaned (1).csv': {'region': 'North', 'diet': 'nonveg', 'category': 'regional'},
                    'southveg_cleaned.csv': {'region': 'South', 'diet': 'veg', 'category': 'regional'},
                    'southnonveg_cleaned.csv': {'region': 'South', 'diet': 'nonveg', 'category': 'regional'},
                }
            },
            'data_2': {
                'name': 'Trimester-Wise Diets',
                'files': {
                    'Trimester_Wise_Diet_Plan.csv': {'region': 'All', 'diet': 'all', 'category': 'trimester'},
                    'pregnancy_diet_1st_2nd_3rd_trimester.xlsx.csv': {'region': 'All', 'diet': 'all', 'category': 'trimester'},
                }
            },
            'data_3': {
                'name': 'Seasonal Diets',
                'files': {
                    'monsoon_diet_pregnant_women.csv': {'region': 'All', 'diet': 'all', 'season': 'monsoon', 'category': 'seasonal'},
                    'summer_pregnancy_diet.csv': {'region': 'All', 'diet': 'all', 'season': 'summer', 'category': 'seasonal'},
                    'Winter_Pregnancy_Diet.csv': {'region': 'All', 'diet': 'all', 'season': 'winter', 'category': 'seasonal'},
                }
            },
            'diabetiesdatasets': {
                'name': 'Diabetes-Pregnancy Specific Diets',
                'files': {
                    'diabetes_pregnancy_indian_foods.csv': {'region': 'All', 'diet': 'all', 'condition': 'diabetes', 'category': 'special_condition'},
                    'gestational_diabetes_indian_diet_dataset.csv': {'region': 'All', 'diet': 'all', 'condition': 'gestational_diabetes', 'category': 'special_condition'},
                    'Indian_Diabetes_Diet (1).csv': {'region': 'All', 'diet': 'all', 'condition': 'diabetes', 'category': 'special_condition'},
                }
            },
            'remainingdatasets': {
                'name': 'Specialized Dietary Guidance',
                'files': {
                    'foods_to_avoid_during_pregnancy_dataset.csv': {'region': 'All', 'diet': 'all', 'category': 'guidance', 'type': 'avoid_foods'},
                    'indian_diet_diabetes_pregnancy_dataset.csv': {'region': 'All', 'diet': 'all', 'condition': 'diabetes', 'category': 'special_condition'},
                    'postnatal_diet_india_dataset.csv': {'region': 'All', 'diet': 'all', 'phase': 'postnatal', 'category': 'postpartum'},
                    'postpartum_diet7_structured_dataset.csv': {'region': 'All', 'diet': 'all', 'phase': 'postpartum', 'category': 'postpartum'},
                    'pregnancy_diet_clean_dataset.csv': {'region': 'All', 'diet': 'all', 'category': 'general_pregnancy'},
                    'pregnancy_dos_donts_dataset.csv': {'region': 'All', 'diet': 'all', 'category': 'guidance', 'type': 'dos_donts'},
                }
            }
        }
        
        # Storage structures
        self.meals = []  # All meals loaded
        self.guidance = []  # Dos/Don'ts and foods to avoid
        self.meals_by_category = defaultdict(list)  # Organize by category
        self.loading_stats = {'loaded': 0, 'failed': 0, 'by_folder': {}}
        
        # FAST LOOKUP CACHES
        self.food_index = {}  # Fast food name lookup
        self.keyword_index = defaultdict(list)  # Fast keyword-based lookup
        self.dos_donts_index = {}  # Fast dos/donts lookup
        self.lock = threading.Lock()  # Thread safety for caching
        
        # Meal preference cache for faster repeated queries
        self._preference_cache = {}
        self._preference_cache_max_size = 100  # Limit cache size
        
        # Load all datasets
        self._load_all_datasets()
        
        # Build fast lookup indexes
        try:
            self._build_fast_indexes()
        except Exception:
            pass  # Silently handle index building errors
        
        self._print_loading_stats()
    
    def _load_all_datasets(self):
        """Load all datasets from all 5 data folders."""
        for folder_name, folder_config in self.dataset_configs.items():
            folder_path = os.path.join(self.base_dir, folder_name)
            self.loading_stats['by_folder'][folder_name] = {'loaded': 0, 'failed': 0}
            
            if not os.path.exists(folder_path):
                # Silently skip missing folders
                continue
            
            # Silently load data
            
            for filename, file_config in folder_config['files'].items():
                file_path = os.path.join(folder_path, filename)
                
                if os.path.exists(file_path):
                    try:
                        df = self._load_csv_file(file_path)
                        
                        if df is not None and len(df) > 0:
                            # Add metadata to rows
                            for col_name, col_value in file_config.items():
                                df[f'source_{col_name}'] = col_value
                            
                            # Categorize based on type
                            if file_config.get('type') in ['dos_donts', 'avoid_foods']:
                                self.guidance.extend(df.to_dict('records'))
                            else:
                                self.meals.extend(df.to_dict('records'))
                                self.meals_by_category[file_config['category']].extend(df.to_dict('records'))
                            
                            count = len(df)
                            self.loading_stats['loaded'] += count
                            self.loading_stats['by_folder'][folder_name]['loaded'] += count
                            # Suppress verbose output
                            pass
                        else:
                            self.loading_stats['by_folder'][folder_name]['failed'] += 1
                    except Exception as e:
                        self.loading_stats['by_folder'][folder_name]['failed'] += 1
                        # Suppress error output
                else:
                    self.loading_stats['by_folder'][folder_name]['failed'] += 1
                    # Suppress file not found messages
    
    def _load_csv_file(self, file_path: str) -> Optional[pd.DataFrame]:
        """Load CSV file with multiple encoding support."""
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1', 'ascii']
        
        for encoding in encodings:
            try:
                df = pd.read_csv(file_path, encoding=encoding, on_bad_lines='skip')
                
                # Clean column names
                df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]
                
                # Remove completely empty rows
                df = df.dropna(how='all')
                
                return df if len(df) > 0 else None
            except Exception:
                continue
        
        return None
    
    def _print_loading_stats(self):
        """Print dataset loading statistics."""
        # Suppress verbose statistics output
        pass
    
    def get_meals_by_preference(self, 
                                region: Optional[str] = None,
                                diet_type: Optional[str] = None,
                                trimester: Optional[int] = None,
                                season: Optional[str] = None,
                                condition: Optional[str] = None,
                                meal_type: Optional[str] = None,
                                log_relaxation: bool = False) -> List[Dict]:
        """Get meals matching specified preferences with progressive filter relaxation.
        
        Args:
            region: 'North' or 'South'
            diet_type: 'veg', 'nonveg', 'vegan'
            trimester: 1, 2, or 3
            season: 'summer', 'winter', 'monsoon'
            condition: 'diabetes', 'gestational_diabetes', etc.
            meal_type: 'breakfast', 'lunch', 'dinner', 'snack'
            log_relaxation: If True, logs which filters were relaxed
        
        Returns:
            List of matching meals
        """
        # Try with all filters first
        results = self._search_with_filters(region, diet_type, trimester, season, condition, meal_type)
        
        if results:
            return results
        
        # Progressive relaxation: relax one filter at a time
        relaxed_filters = []
        
        # Step 1: Try without season (most specific, least critical)
        if season:
            results = self._search_with_filters(region, diet_type, trimester, None, condition, meal_type)
            if results:
                if log_relaxation:
                    relaxed_filters.append('season')
                    print(f"ℹ️ Relaxed filter: season (was: {season})")
                return results
        
        # Step 2: Try without condition
        if condition:
            results = self._search_with_filters(region, diet_type, trimester, None, None, meal_type)
            if results:
                if log_relaxation:
                    if season:
                        relaxed_filters.append('season')
                    relaxed_filters.append('condition')
                    print(f"ℹ️ Relaxed filters: {', '.join(f for f in relaxed_filters if f)}")
                return results
        
        # Step 3: Try without trimester
        if trimester:
            results = self._search_with_filters(region, diet_type, None, None, None, meal_type)
            if results:
                if log_relaxation:
                    if season:
                        relaxed_filters.append('season')
                    if condition:
                        relaxed_filters.append('condition')
                    relaxed_filters.append('trimester')
                    print(f"ℹ️ Relaxed filters: {', '.join(f for f in relaxed_filters if f)}")
                return results
        
        # Step 4: Try without meal_type
        if meal_type:
            results = self._search_with_filters(region, diet_type, None, None, None, None)
            if results:
                if log_relaxation:
                    if season:
                        relaxed_filters.append('season')
                    if condition:
                        relaxed_filters.append('condition')
                    if trimester:
                        relaxed_filters.append('trimester')
                    relaxed_filters.append('meal_type')
                    print(f"ℹ️ Relaxed filters: {', '.join(f for f in relaxed_filters if f)}")
                return results
        
        # Final fallback: just region and diet (never relax these)
        results = self._search_with_filters(region, diet_type, None, None, None, None)
        if log_relaxation and results:
            print(f"ℹ️ Using only region and diet_type filters")
        
        return results
    
    def _search_with_filters(self,
                            region: Optional[str] = None,
                            diet_type: Optional[str] = None,
                            trimester: Optional[int] = None,
                            season: Optional[str] = None,
                            condition: Optional[str] = None,
                            meal_type: Optional[str] = None) -> List[Dict]:
        """Internal method to search meals with specific filter combination."""
        # Create cache key
        cache_key = f"{region}_{diet_type}_{trimester}_{season}_{condition}_{meal_type}"
        
        # Check cache first
        if cache_key in self._preference_cache:
            return self._preference_cache[cache_key]
        
        # Normalize inputs
        results = []
        normalized_region = self._normalize_region(region)
        normalized_diet = self._normalize_diet(diet_type)
        normalized_season = self._normalize_season(season)
        normalized_condition = self._normalize_condition(condition)
        normalized_meal_type = self._normalize_meal_type(meal_type)
        
        for meal in self.meals:
            match = True
            
            # Check regional preference
            if normalized_region:
                source_region = meal.get('source_region')
                if source_region:
                    source_region_lower = source_region.lower()
                    if source_region_lower != normalized_region and source_region_lower != 'all':
                        match = False
            
            # Check diet type
            if normalized_diet:
                source_diet = meal.get('source_diet')
                if source_diet:
                    source_diet_lower = source_diet.lower()
                    if source_diet_lower != normalized_diet and source_diet_lower != 'all':
                        match = False
            
            # Check meal type
            if normalized_meal_type:
                meal_col = self._find_meal_type_column(meal)
                if meal_col:
                    if meal.get(meal_col, '').lower() != normalized_meal_type:
                        match = False
            
            # Check special conditions
            if normalized_condition:
                source_condition = meal.get('source_condition')
                if source_condition:
                    if source_condition.lower() != normalized_condition:
                        match = False
            
            # Check season
            if normalized_season:
                source_season = meal.get('source_season')
                if source_season:
                    source_season_lower = source_season.lower()
                    if source_season_lower != normalized_season and source_season_lower != 'all':
                        match = False
            
            # Check trimester
            if trimester:
                trimester_col = self._find_trimester_column(meal)
                if trimester_col:
                    meal_trimester = meal.get(trimester_col)
                    if meal_trimester and str(trimester) not in str(meal_trimester):
                        match = False
            
            if match:
                results.append(meal)
        
        # Store in cache (limit cache size)
        if len(self._preference_cache) < self._preference_cache_max_size:
            self._preference_cache[cache_key] = results
        
        return results

    def _find_meal_type_column(self, meal: Dict) -> Optional[str]:
        """Find the column containing meal type information."""
        possible_columns = ['meal_type', 'type', 'meal', 'breakfast_lunch_dinner', 'meal_time']
        for col in possible_columns:
            if col in meal:
                return col
        return None
    
    def _find_trimester_column(self, meal: Dict) -> Optional[str]:
        """Find the column containing trimester information."""
        possible_columns = ['trimester', 'trimester_wise', 'month', 'week']
        for col in possible_columns:
            if col in meal:
                return col
        return None

    def _normalize_region(self, region: Optional[str]) -> Optional[str]:
        if not region:
            return None
        value = region.strip().lower()
        if 'north' in value:
            return 'north'
        if 'south' in value:
            return 'south'
        return value if value not in ['all', 'any'] else None

    def _normalize_diet(self, diet: Optional[str]) -> Optional[str]:
        if not diet:
            return None
        value = diet.strip().lower()
        if 'veg' in value and 'non' not in value:
            return 'veg'
        if 'non' in value:
            return 'nonveg'
        if 'vegan' in value:
            return 'vegan'
        return value if value not in ['mixed', 'all', 'any'] else None

    def _normalize_season(self, season: Optional[str]) -> Optional[str]:
        if not season:
            return None
        value = season.strip().lower()
        if 'summer' in value:
            return 'summer'
        if 'winter' in value:
            return 'winter'
        if 'monsoon' in value or 'rain' in value:
            return 'monsoon'
        return value if value not in ['all', 'any'] else None

    def _normalize_condition(self, condition: Optional[str]) -> Optional[str]:
        if not condition:
            return None
        value = condition.strip().lower()
        if 'gestational' in value:
            return 'gestational_diabetes'
        if 'diabetes' in value:
            return 'diabetes'
        return value if value not in ['general', 'none'] else None

    def _normalize_meal_type(self, meal_type: Optional[str]) -> Optional[str]:
        if not meal_type:
            return None
        value = meal_type.strip().lower()
        if value in ['snack', 'snacks']:
            return 'snack'
        return value
    
    def get_guidance(self, guidance_type: Optional[str] = None) -> List[Dict]:
        """Get guidance items (dos/donts, foods to avoid).
        
        Args:
            guidance_type: 'dos_donts' or 'avoid_foods', None for all
        
        Returns:
            List of guidance items
        """
        if guidance_type is None:
            return self.guidance
        
        return [g for g in self.guidance if g.get('source_type', '').lower() == guidance_type.lower()]
    
    def get_meals_for_condition(self, condition: str) -> List[Dict]:
        """Get meals suitable for specific health conditions."""
        return self.get_meals_by_preference(condition=condition)
    
    def get_seasonal_meals(self, season: str) -> List[Dict]:
        """Get meals recommended for a specific season."""
        return self.get_meals_by_preference(season=season)
    
    def get_trimester_meals(self, trimester: int) -> List[Dict]:
        """Get meals recommended for a specific trimester."""
        return self.get_meals_by_preference(trimester=trimester)
    
    def get_regional_meals(self, region: str) -> List[Dict]:
        """Get meals for a specific region."""
        return self.get_meals_by_preference(region=region)
    
    def get_statistics(self) -> Dict:
        """Get dataset statistics."""
        return {
            'total_meals': len(self.meals),
            'total_guidance': len(self.guidance),
            'categories': {cat: len(meals) for cat, meals in self.meals_by_category.items()},
            'by_region': self._count_by_field('source_region'),
            'by_diet': self._count_by_field('source_diet'),
            'by_condition': self._count_by_field('source_condition'),
            'by_season': self._count_by_field('source_season'),
        }
    
    def _count_by_field(self, field_name: str) -> Dict[str, int]:
        """Count meals by a specific field."""
        counts = defaultdict(int)
        for meal in self.meals:
            value = meal.get(field_name, 'unknown')
            if value:
                counts[value] += 1
        return dict(counts)
    
    def get_available_options(self) -> Dict[str, List[str]]:
        """Get all available options for user preference selection.
        
        Filters out generic values like 'All' and 'all' to show only specific options.
        """
        stats = self.get_statistics()
        return {
            # Only show specific regions (North, South), not 'All'
            'regions': sorted([r for r in stats['by_region'].keys() if r and r.lower() != 'all']),
            # Only show specific diets (veg, nonveg), not 'all'
            'diets': sorted([d for d in stats['by_diet'].keys() if d and d.lower() != 'all']),
            # Show all seasons
            'conditions': sorted([c for c in stats['by_condition'].keys() if c]),
            # Show all seasons
            'seasons': sorted([s for s in stats['by_season'].keys() if s]),
            'categories': list(self.meals_by_category.keys()),
        }
    
    def get_nutritional_data(self, meal_name: str) -> Dict:
        """
        Look up nutritional data for a specific meal from the datasets.
        
        Args:
            meal_name: Name of the meal to look up
            
        Returns:
            Dictionary with estimated nutritional values:
            {
                'calories': float,
                'protein': float,
                'carbs': float,
                'fat': float,
                'iron': float,
                'calcium': float,
                'folic_acid': float,
                'fiber': float,
                'vitamin_a': float,
                'vitamin_c': float
            }
            Returns empty dict if no data found
        """
        meal_name_lower = meal_name.strip().lower()
        
        # Search in meals for matching name
        for meal in self.meals:
            # Try various name columns
            for name_col in ['food', 'food_item', 'meal', 'dish', 'dish_name', 'name', 'item', 'recipe']:
                if name_col in meal and meal[name_col]:
                    if meal[name_col].strip().lower() == meal_name_lower:
                        # Extract nutritional data from this meal
                        nutrition = {}
                        
                        # Map common nutrition column names to standard keys
                        nutrient_mapping = {
                            'calories': ['calories', 'calorie', 'energy', 'kcal'],
                            'protein': ['protein', 'proteins'],
                            'carbs': ['carbohydrate', 'carbs', 'carbohydrates'],
                            'fat': ['fat', 'fats', 'total_fat'],
                            'iron': ['iron', 'fe'],
                            'calcium': ['calcium', 'ca'],
                            'folic_acid': ['folic_acid', 'folate', 'folic'],
                            'fiber': ['fiber', 'dietary_fiber', 'fibre'],
                            'vitamin_a': ['vitamin_a', 'vit_a', 'retinol'],
                            'vitamin_c': ['vitamin_c', 'vit_c', 'ascorbic_acid'],
                            'vitamin_b6': ['vitamin_b6', 'vit_b6', 'pyridoxine'],
                            'vitamin_b12': ['vitamin_b12', 'vit_b12', 'cobalamin'],
                            'vitamin_d': ['vitamin_d', 'vit_d'],
                            'zinc': ['zinc', 'zn'],
                            'magnesium': ['magnesium', 'mg']
                        }
                        
                        # Extract nutrients
                        for nutrient_key, possible_cols in nutrient_mapping.items():
                            for col in possible_cols:
                                if col in meal and meal[col]:
                                    try:
                                        # Try to convert to float, removing units
                                        value_str = str(meal[col]).lower().replace('g', '').replace('mg', '').replace('mcg', '').replace('kcal', '').strip()
                                        nutrition[nutrient_key] = float(value_str)
                                        break
                                    except (ValueError, TypeError):
                                        pass
                        
                        return nutrition
        
        # If exact match not found, return default estimates based on meal name keywords
        return self._estimate_nutrition_from_keywords(meal_name)
    
    def _estimate_nutrition_from_keywords(self, meal_name: str) -> Dict:
        """
        Estimate basic nutrition from meal name keywords when exact data not available.
        
        Args:
            meal_name: Name of the meal
            
        Returns:
            Dictionary with estimated nutritional values
        """
        meal_lower = meal_name.lower()
        
        # Default baseline nutrition (per serving)
        nutrition = {
            'calories': 300.0,
            'protein': 10.0,
            'carbs': 40.0,
            'fat': 8.0,
            'iron': 2.0,
            'calcium': 100.0,
            'folic_acid': 50.0,
            'fiber': 5.0
        }
        
        # Adjust based on keywords
        # High protein foods
        if any(word in meal_lower for word in ['egg', 'chicken', 'fish', 'dal', 'lentil', 'paneer', 'tofu', 'meat']):
            nutrition['protein'] += 15
            nutrition['calories'] += 50
            nutrition['iron'] += 2
        
        # High carb foods
        if any(word in meal_lower for word in ['rice', 'roti', 'bread', 'pasta', 'chapati', 'paratha']):
            nutrition['carbs'] += 20
            nutrition['calories'] += 100
        
        # Dairy
        if any(word in meal_lower for word in ['milk', 'yogurt', 'curd', 'cheese', 'paneer']):
            nutrition['calcium'] += 150
            nutrition['protein'] += 8
        
        # Leafy greens
        if any(word in meal_lower for word in ['spinach', 'palak', 'methi', 'kale', 'fenugreek']):
            nutrition['iron'] += 3
            nutrition['folic_acid'] += 100
            nutrition['fiber'] += 3
            nutrition['calcium'] += 50
        
        # Fruits
        if any(word in meal_lower for word in ['fruit', 'apple', 'banana', 'orange', 'mango', 'papaya']):
            nutrition['fiber'] += 3
            nutrition['folic_acid'] += 30
            nutrition['carbs'] += 15
        
        return nutrition
