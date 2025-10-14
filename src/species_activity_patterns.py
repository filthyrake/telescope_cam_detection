"""
Species Activity Pattern Database
Maps species to their typical activity patterns (diurnal, nocturnal, crepuscular).
Used for filtering Stage 2 species classification results based on time of day.
"""

from typing import Dict, Set
from enum import Enum


class ActivityPattern(Enum):
    """Species activity patterns."""
    DIURNAL = "diurnal"  # Active during day
    NOCTURNAL = "nocturnal"  # Active at night
    CREPUSCULAR = "crepuscular"  # Active at dawn/dusk
    CATHEMERAL = "cathemeral"  # Active any time


# Comprehensive species activity database
# Format: species_name → ActivityPattern
SPECIES_ACTIVITY_PATTERNS: Dict[str, ActivityPattern] = {
    # === MAMMALS ===

    # Bats (all nocturnal)
    "Mexican Free-tailed Bat": ActivityPattern.NOCTURNAL,
    "Pallid Bat": ActivityPattern.NOCTURNAL,
    "California Leaf-nosed Bat": ActivityPattern.NOCTURNAL,
    "Big Brown Bat": ActivityPattern.NOCTURNAL,
    "Hoary Bat": ActivityPattern.NOCTURNAL,
    "Silver-haired Bat": ActivityPattern.NOCTURNAL,

    # Canids (mostly crepuscular/nocturnal)
    "Coyote": ActivityPattern.CREPUSCULAR,
    "Kit Fox": ActivityPattern.NOCTURNAL,
    "Gray Fox": ActivityPattern.CREPUSCULAR,

    # Felids (mostly crepuscular/nocturnal)
    "Bobcat": ActivityPattern.CREPUSCULAR,
    "Mountain Lion": ActivityPattern.CREPUSCULAR,
    "Cougar": ActivityPattern.CREPUSCULAR,
    "Domestic Cat": ActivityPattern.CATHEMERAL,

    # Rabbits (crepuscular)
    "Desert Cottontail": ActivityPattern.CREPUSCULAR,
    "Black-tailed Jackrabbit": ActivityPattern.CREPUSCULAR,
    "Antelope Jackrabbit": ActivityPattern.CREPUSCULAR,

    # Rodents (mostly nocturnal to crepuscular)
    "White-tailed Antelope Squirrel": ActivityPattern.DIURNAL,  # Exception: diurnal
    "Harris's Antelope Squirrel": ActivityPattern.DIURNAL,  # Exception: diurnal
    "Round-tailed Ground Squirrel": ActivityPattern.DIURNAL,
    "Rock Squirrel": ActivityPattern.DIURNAL,
    "Desert Woodrat": ActivityPattern.NOCTURNAL,
    "Merriam's Kangaroo Rat": ActivityPattern.NOCTURNAL,
    "Desert Kangaroo Rat": ActivityPattern.NOCTURNAL,
    "Desert Pocket Mouse": ActivityPattern.NOCTURNAL,
    "Cactus Mouse": ActivityPattern.NOCTURNAL,
    "Southern Grasshopper Mouse": ActivityPattern.NOCTURNAL,

    # Ungulates (crepuscular)
    "Mule Deer": ActivityPattern.CREPUSCULAR,
    "Collared Peccary": ActivityPattern.CREPUSCULAR,
    "Javelina": ActivityPattern.CREPUSCULAR,
    "Desert Bighorn Sheep": ActivityPattern.CREPUSCULAR,

    # Other mammals
    "Ringtail": ActivityPattern.NOCTURNAL,
    "American Badger": ActivityPattern.CREPUSCULAR,
    "Striped Skunk": ActivityPattern.NOCTURNAL,
    "Western Spotted Skunk": ActivityPattern.NOCTURNAL,
    "Desert Shrew": ActivityPattern.CATHEMERAL,
    "Domestic Dog": ActivityPattern.CATHEMERAL,

    # === BIRDS ===

    # Owls (nocturnal - exception to diurnal birds)
    "Great Horned Owl": ActivityPattern.NOCTURNAL,
    "Barn Owl": ActivityPattern.NOCTURNAL,
    "Burrowing Owl": ActivityPattern.CREPUSCULAR,  # Active dawn/dusk
    "Elf Owl": ActivityPattern.NOCTURNAL,
    "Western Screech-Owl": ActivityPattern.NOCTURNAL,

    # All other birds (diurnal by default)
    "Gambel's Quail": ActivityPattern.DIURNAL,
    "Greater Roadrunner": ActivityPattern.DIURNAL,
    "Cactus Wren": ActivityPattern.DIURNAL,
    "Curve-billed Thrasher": ActivityPattern.DIURNAL,
    "Le Conte's Thrasher": ActivityPattern.DIURNAL,
    "Bendire's Thrasher": ActivityPattern.DIURNAL,
    "Red-tailed Hawk": ActivityPattern.DIURNAL,
    "Harris's Hawk": ActivityPattern.DIURNAL,
    "Cooper's Hawk": ActivityPattern.DIURNAL,
    "Sharp-shinned Hawk": ActivityPattern.DIURNAL,
    "Golden Eagle": ActivityPattern.DIURNAL,
    "Turkey Vulture": ActivityPattern.DIURNAL,
    "Black Vulture": ActivityPattern.DIURNAL,
    "Common Raven": ActivityPattern.DIURNAL,
    "Chihuahuan Raven": ActivityPattern.DIURNAL,
    "American Crow": ActivityPattern.DIURNAL,
    "Mourning Dove": ActivityPattern.DIURNAL,
    "White-winged Dove": ActivityPattern.DIURNAL,
    "Inca Dove": ActivityPattern.DIURNAL,
    "Common Ground Dove": ActivityPattern.DIURNAL,
    "Gila Woodpecker": ActivityPattern.DIURNAL,
    "Ladder-backed Woodpecker": ActivityPattern.DIURNAL,
    "Gilded Flicker": ActivityPattern.DIURNAL,
    "Northern Flicker": ActivityPattern.DIURNAL,
    "Verdin": ActivityPattern.DIURNAL,
    "Black-throated Sparrow": ActivityPattern.DIURNAL,
    "White-crowned Sparrow": ActivityPattern.DIURNAL,
    "House Finch": ActivityPattern.DIURNAL,
    "Lesser Goldfinch": ActivityPattern.DIURNAL,
    "Phainopepla": ActivityPattern.DIURNAL,
    "Loggerhead Shrike": ActivityPattern.DIURNAL,
    "Northern Mockingbird": ActivityPattern.DIURNAL,
    "Costa's Hummingbird": ActivityPattern.DIURNAL,
    "Anna's Hummingbird": ActivityPattern.DIURNAL,
    "Black-chinned Hummingbird": ActivityPattern.DIURNAL,
    "Rufous Hummingbird": ActivityPattern.DIURNAL,
    "Rock Wren": ActivityPattern.DIURNAL,
    "Canyon Wren": ActivityPattern.DIURNAL,
    "Black-tailed Gnatcatcher": ActivityPattern.DIURNAL,
    "Blue-gray Gnatcatcher": ActivityPattern.DIURNAL,
    "Say's Phoebe": ActivityPattern.DIURNAL,
    "Ash-throated Flycatcher": ActivityPattern.DIURNAL,
    "Vermilion Flycatcher": ActivityPattern.DIURNAL,
    "Horned Lark": ActivityPattern.DIURNAL,

    # === REPTILES ===

    # Lizards (mostly diurnal - need warmth)
    "Desert Iguana": ActivityPattern.DIURNAL,
    "Common Chuckwalla": ActivityPattern.DIURNAL,
    "Chuckwalla": ActivityPattern.DIURNAL,
    "Desert Spiny Lizard": ActivityPattern.DIURNAL,
    "Clark's Spiny Lizard": ActivityPattern.DIURNAL,
    "Zebra-tailed Lizard": ActivityPattern.DIURNAL,
    "Greater Earless Lizard": ActivityPattern.DIURNAL,
    "Desert Horned Lizard": ActivityPattern.DIURNAL,
    "Flat-tailed Horned Lizard": ActivityPattern.DIURNAL,
    "Long-nosed Leopard Lizard": ActivityPattern.DIURNAL,
    "Collared Lizard": ActivityPattern.DIURNAL,
    "Desert Collared Lizard": ActivityPattern.DIURNAL,
    "Common Side-blotched Lizard": ActivityPattern.DIURNAL,
    "Desert Night Lizard": ActivityPattern.NOCTURNAL,  # Exception

    # Geckos (nocturnal - exception to diurnal reptiles)
    "Western Banded Gecko": ActivityPattern.NOCTURNAL,
    "Desert Banded Gecko": ActivityPattern.NOCTURNAL,
    "Mediterranean Gecko": ActivityPattern.NOCTURNAL,

    # Snakes (variable - some diurnal, some nocturnal)
    "Western Diamondback Rattlesnake": ActivityPattern.CREPUSCULAR,
    "Mojave Rattlesnake": ActivityPattern.CREPUSCULAR,
    "Sidewinder": ActivityPattern.NOCTURNAL,
    "Speckled Rattlesnake": ActivityPattern.CREPUSCULAR,
    "Gopher Snake": ActivityPattern.DIURNAL,
    "Gophersnake": ActivityPattern.DIURNAL,
    "Common Kingsnake": ActivityPattern.DIURNAL,
    "California Kingsnake": ActivityPattern.DIURNAL,
    "Long-nosed Snake": ActivityPattern.NOCTURNAL,
    "Coachwhip": ActivityPattern.DIURNAL,
    "Red Coachwhip": ActivityPattern.DIURNAL,
    "Glossy Snake": ActivityPattern.NOCTURNAL,
    "Western Patch-nosed Snake": ActivityPattern.DIURNAL,

    # Tortoises (diurnal)
    "Desert Tortoise": ActivityPattern.DIURNAL,
    "Mohave Desert Tortoise": ActivityPattern.DIURNAL,

    # === AMPHIBIANS ===

    # Toads (mostly nocturnal)
    "Couch's Spadefoot": ActivityPattern.NOCTURNAL,
    "Great Basin Spadefoot": ActivityPattern.NOCTURNAL,
    "Red-spotted Toad": ActivityPattern.NOCTURNAL,
    "Sonoran Desert Toad": ActivityPattern.NOCTURNAL,
    "Colorado River Toad": ActivityPattern.NOCTURNAL,

    # === ARTHROPODS ===

    # Scorpions and spiders (nocturnal)
    "Desert Hairy Scorpion": ActivityPattern.NOCTURNAL,
    "Arizona Bark Scorpion": ActivityPattern.NOCTURNAL,
    "Desert Blonde Tarantula": ActivityPattern.NOCTURNAL,
    "Tarantula": ActivityPattern.NOCTURNAL,

    # === GENERIC/FALLBACK ===
    "Human": ActivityPattern.CATHEMERAL,
    "Person": ActivityPattern.CATHEMERAL,
}


def get_species_activity(species_name: str) -> ActivityPattern:
    """
    Get activity pattern for a species.

    Args:
        species_name: Species common name

    Returns:
        ActivityPattern for the species, or CATHEMERAL if unknown
    """
    # Try exact match first
    pattern = SPECIES_ACTIVITY_PATTERNS.get(species_name)
    if pattern:
        return pattern

    # Try case-insensitive match
    species_lower = species_name.lower()
    for known_species, activity in SPECIES_ACTIVITY_PATTERNS.items():
        if known_species.lower() == species_lower:
            return activity

    # Default fallback based on broad categories
    species_lower = species_name.lower()

    # Owls are nocturnal
    if 'owl' in species_lower:
        return ActivityPattern.NOCTURNAL

    # Bats are nocturnal
    if 'bat' in species_lower and 'combat' not in species_lower:
        return ActivityPattern.NOCTURNAL

    # Most birds are diurnal
    if any(bird_word in species_lower for bird_word in
           ['bird', 'hawk', 'eagle', 'raven', 'crow', 'dove', 'quail',
            'wren', 'sparrow', 'finch', 'hummingbird', 'woodpecker']):
        return ActivityPattern.DIURNAL

    # Geckos are nocturnal
    if 'gecko' in species_lower:
        return ActivityPattern.NOCTURNAL

    # Most lizards are diurnal
    if 'lizard' in species_lower:
        return ActivityPattern.DIURNAL

    # Most scorpions/spiders are nocturnal
    if any(word in species_lower for word in ['scorpion', 'tarantula', 'spider']):
        return ActivityPattern.NOCTURNAL

    # Unknown species - assume cathemeral (can be active anytime)
    return ActivityPattern.CATHEMERAL


def get_diurnal_species() -> Set[str]:
    """Get set of all diurnal species."""
    return {species for species, pattern in SPECIES_ACTIVITY_PATTERNS.items()
            if pattern == ActivityPattern.DIURNAL}


def get_nocturnal_species() -> Set[str]:
    """Get set of all nocturnal species."""
    return {species for species, pattern in SPECIES_ACTIVITY_PATTERNS.items()
            if pattern == ActivityPattern.NOCTURNAL}


def get_crepuscular_species() -> Set[str]:
    """Get set of all crepuscular species."""
    return {species for species, pattern in SPECIES_ACTIVITY_PATTERNS.items()
            if pattern == ActivityPattern.CREPUSCULAR}


def is_species_likely_active(species_name: str, time_of_day: str) -> bool:
    """
    Check if a species is likely to be active at a given time of day.

    Args:
        species_name: Species common name
        time_of_day: Time category ('dawn', 'day', 'dusk', 'night')

    Returns:
        True if species is likely active at this time
    """
    pattern = get_species_activity(species_name)

    if pattern == ActivityPattern.CATHEMERAL:
        return True  # Active any time
    elif pattern == ActivityPattern.DIURNAL:
        return time_of_day in ['dawn', 'day', 'dusk']
    elif pattern == ActivityPattern.NOCTURNAL:
        return time_of_day in ['dusk', 'night', 'dawn']
    elif pattern == ActivityPattern.CREPUSCULAR:
        return time_of_day in ['dawn', 'dusk', 'night']

    return True  # Default to allowing


if __name__ == "__main__":
    # Test the species activity database
    print("=== Species Activity Pattern Database ===\n")

    test_cases = [
        ("Great Horned Owl", "night"),
        ("Gambel's Quail", "day"),
        ("Coyote", "dusk"),
        ("Mexican Free-tailed Bat", "night"),
        ("Desert Iguana", "day"),
        ("Western Banded Gecko", "night"),
        ("Unknown Bird Species", "night"),
    ]

    for species, time in test_cases:
        activity = get_species_activity(species)
        is_active = is_species_likely_active(species, time)
        print(f"{species:30} @ {time:6} → {activity.value:12} (active: {is_active})")

    print(f"\n✓ Database contains {len(SPECIES_ACTIVITY_PATTERNS)} species")
    print(f"  - Diurnal: {len(get_diurnal_species())}")
    print(f"  - Nocturnal: {len(get_nocturnal_species())}")
    print(f"  - Crepuscular: {len(get_crepuscular_species())}")
