"""Dataset loader for meal planning from CSV files."""
import os
import csv
import pandas as pd
from typing import List, Dict, Optional
from collections import defaultdict


class DatasetLoader:
    """Load and manage meal plan datasets from CSV files."""
    
    def __init__(self, data_dir: str = 'data/data_1'):
        """Initialize dataset loader."""
        self.data_dir = data_dir
        self.meals_by_region = {}
        self.meals_by_diet = {}
        self.all_meals = []
        self.loading_stats = {'loaded': 0, 'failed': 0}
        self._load_datasets()
        self._print_loading_stats()
    
    def _load_datasets(self):
        """Load all meal plan datasets from CSV files."""
        # Define dataset files with region and diet type info
        datasets = {
            'northveg_cleaned.csv': ('North Indian', 'veg'),
            'northnonveg_cleaned.csv': ('North Indian', 'nonveg'),
            'southveg_cleaned.csv': ('South Indian', 'veg'),
            'southnonveg_cleaned.csv': ('South Indian', 'nonveg'),
        }
        
        for filename, (region, diet) in datasets.items():
            filepath = os.path.join(self.data_dir, filename)
            if os.path.exists(filepath):
                self._load_csv_file(filepath, region, diet)
            else:
                self.loading_stats['failed'] += 1
    
    def _load_csv_file(self, filepath: str, region: str, diet_type: str):
        """Load a single CSV file with proper encoding and validation."""
        try:
            # Try multiple encodings
            encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'iso-8859-1', 'cp1252']
            df = None
            
            for encoding in encodings:
                try:
                    df = pd.read_csv(filepath, encoding=encoding)
                    break
                except (UnicodeDecodeError, UnicodeError, pd.errors.ParserError):
                    continue
            
            if df is None or len(df) == 0:
                print(f"Warning: Could not load {os.path.basename(filepath)}")
                self.loading_stats['failed'] += 1
                return
            
            # Process each row
            valid_entries = 0
            for idx, row in df.iterrows():
                # Get food item with flexible column names
                food = self._get_value(row, ['Food', 'food', 'Food_Item', 'FOOD']).strip()
                
                if not food or food.lower() in ['nan', 'none', '']:
                    continue
                
                # Get meal type
                meal_type = self._get_value(row, ['Meal Type', 'meal_type', 'MealType', 'meal type']).strip()
                if not meal_type:
                    meal_type = self._infer_meal_type(food)
                
                # Get day info
                day = self._get_value(row, ['Day', 'day', 'DAY']).strip()
                
                # Get trimester info
                trimester = self._get_value(row, ['Trimester', 'trimester', 'TRIMESTER'])
                if not trimester:
                    trimester = 'All Trimesters'
                else:
                    trimester = trimester.strip()
                
                meal_entry = {
                    'day': day,
                    'meal_type': meal_type,
                    'food': food,
                    'regional_type': region,
                    'food_type': diet_type,
                    'trimester': trimester,
                    'region': region,
                    'diet': diet_type
                }
                
                # Add to collections
                self.all_meals.append(meal_entry)
                
                # Index by region
                if region not in self.meals_by_region:
                    self.meals_by_region[region] = []
                self.meals_by_region[region].append(meal_entry)
                
                # Index by diet type
                diet_key = 'vegetarian' if diet_type.lower() in ['veg', 'vegetarian'] else 'non-vegetarian'
                if diet_key not in self.meals_by_diet:
                    self.meals_by_diet[diet_key] = []
                self.meals_by_diet[diet_key].append(meal_entry)
                
                valid_entries += 1
            
            self.loading_stats['loaded'] += 1
            print(f"✓ Loaded: {os.path.basename(filepath)} ({valid_entries} meals) [{region}, {diet_type}]")
            
        except Exception as e:
            print(f"✗ Error loading {filepath}: {e}")
            self.loading_stats['failed'] += 1
    
    def _get_value(self, row, possible_columns):
        """Get value from row using possible column names."""
        for col in possible_columns:
            if col in row.index:
                value = row[col]
                if pd.notna(value):
                    return str(value)
        return ''
    
    def _infer_meal_type(self, food_name: str) -> str:
        """Infer meal type from food name if not specified."""
        food_lower = food_name.lower()
        
        if any(word in food_lower for word in ['tea', 'coffee', 'milk', 'juice', 'juice', 'cereal', 'toast', 'eggs', 'bread']):
            return 'breakfast'
        elif any(word in food_lower for word in ['rice', 'curry', 'dal', 'sabzi', 'sambar', 'rasam']):
            return 'lunch'
        elif any(word in food_lower for word in ['roti', 'dal', 'sabzi', 'khichdi', 'soup']):
            return 'dinner'
        elif any(word in food_lower for word in ['fruit', 'nut', 'biscuit', 'snack', 'cake']):
            return 'snack'
        else:
            return 'lunch'  # Default to lunch
    
    def _print_loading_stats(self):
        """Print loading statistics."""
        total_meals = len(self.all_meals)
        print(f"\n{'='*70}")
        print(f"DATASET LOADING SUMMARY")
        print(f"{'='*70}")
        print(f"✓ Total meals loaded: {total_meals}")
        print(f"✓ Files loaded: {self.loading_stats['loaded']}")
        if self.loading_stats['failed'] > 0:
            print(f"✗ Files failed: {self.loading_stats['failed']}")
        if self.meals_by_region:
            print(f"✓ Regions: {', '.join(self.meals_by_region.keys())}")
        if self.meals_by_diet:
            print(f"✓ Diet types: {', '.join(self.meals_by_diet.keys())}")
        print(f"{'='*70}\n")
    
    def get_meals_for_meal_type(
        self,
        meal_type: str,
        region: Optional[str] = None,
        diet_type: Optional[str] = None,
        trimester: Optional[str] = None
    ) -> List[Dict]:
        """Get meals matching criteria with improved filtering."""
        filtered = self.all_meals
        
        if not filtered:
            return []
        
        # Filter by meal type
        if meal_type:
            meal_type_lower = meal_type.lower().replace('_', ' ')
            filtered = [
                m for m in filtered 
                if self._match_meal_type(m['meal_type'].lower(), meal_type_lower)
            ]
        
        # Filter by region
        if region:
            region_lower = region.lower()
            filtered = [
                m for m in filtered
                if region_lower in m['regional_type'].lower() or 
                   (region.replace(' ', '').lower() in m['regional_type'].replace(' ', '').lower())
            ]
        
        # Filter by diet type
        if diet_type:
            diet_lower = diet_type.lower()
            if diet_lower in ['vegetarian', 'veg']:
                filtered = [m for m in filtered if m['food_type'].lower() in ['veg', 'vegetarian']]
            elif diet_lower in ['non-vegetarian', 'non_vegetarian', 'nonveg', 'non-veg']:
                filtered = [m for m in filtered if m['food_type'].lower() in ['nonveg', 'non-veg', 'non_veg']]
        
        # Filter by trimester
        if trimester:
            trimester_lower = trimester.lower()
            filtered = [
                m for m in filtered
                if trimester_lower in m['trimester'].lower() or 
                   'all' in m['trimester'].lower()
            ]
        
        return filtered
    
    def _match_meal_type(self, meal_type: str, target: str) -> bool:
        """Check if meal type matches target."""
        meal_type = meal_type.replace('_', ' ')
        target = target.replace('_', ' ')
        
        # Direct match
        if meal_type == target:
            return True
        
        # Snack matching
        if 'snack' in target and 'snack' in meal_type:
            return True
        
        # Partial match
        if target in meal_type or meal_type in target:
            return True
        
        return False
    
    def get_breakfast_options(self, region: Optional[str] = None, diet_type: Optional[str] = None, trimester: Optional[str] = None) -> List[str]:
        """Get breakfast food options."""
        meals = self.get_meals_for_meal_type('breakfast', region, diet_type, trimester)
        return list(set([m['food'] for m in meals if m['food']]))[:15]
    
    def get_lunch_options(self, region: Optional[str] = None, diet_type: Optional[str] = None, trimester: Optional[str] = None) -> List[str]:
        """Get lunch food options."""
        meals = self.get_meals_for_meal_type('lunch', region, diet_type, trimester)
        return list(set([m['food'] for m in meals if m['food']]))[:15]
    
    def get_dinner_options(self, region: Optional[str] = None, diet_type: Optional[str] = None, trimester: Optional[str] = None) -> List[str]:
        """Get dinner food options."""
        meals = self.get_meals_for_meal_type('dinner', region, diet_type, trimester)
        return list(set([m['food'] for m in meals if m['food']]))[:15]
    
    def get_snack_options(self, region: Optional[str] = None, diet_type: Optional[str] = None, trimester: Optional[str] = None) -> List[str]:
        """Get snack food options."""
        meals = self.get_meals_for_meal_type('snack', region, diet_type, trimester)
        snacks = list(set([m['food'] for m in meals if m['food']]))[:10]
        
        # If no snacks found, return some healthy default snacks
        if not snacks:
            snacks = ['Fresh Fruits', 'Nuts and Seeds', 'Yogurt', 'Milk', 'Dry Fruits']
        
        return snacks
    
    def get_statistics(self) -> Dict:
        """Get statistics about loaded datasets."""
        return {
            'total_meals': len(self.all_meals),
            'regions': list(self.meals_by_region.keys()),
            'diet_types': list(self.meals_by_diet.keys()),
            'meals_by_region': {k: len(v) for k, v in self.meals_by_region.items()},
            'meals_by_diet': {k: len(v) for k, v in self.meals_by_diet.items()},
            'loading_status': f"Loaded {self.loading_stats['loaded']} files, {self.loading_stats['failed']} failed"        }