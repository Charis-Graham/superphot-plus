from enum import Enum
import numpy as np

class HierarchyClass(str, Enum):
    """Original Classes of Supernovae"""

    @classmethod
    def get_alternative_namings(cls):
        """Returns the alternative namings for each supernova class.

        Parameters
        ----------
        cls : SupernovaClass
            The SupernovaClass class.

        Returns
        -------
        dict
            A dictionary that maps each supernova class label to its
            respective alternative namings.
        """
        return {
            "SN Ia": [
                "SN Ia-91T-like",
                "SN Ia-CSM",
                "SN Ia-91bg-like",
                "SNIa",
                "SN Ia-91T",
                "SN Ia-91bg",
                "10",
                "11",
                "12",
            ],
            "SN Ib": [
                "SN Ic",
                "SN Ib",
                "SN Ic-BL",
                "SN Ib-Ca-rich",
                "SN Ib/c",
                "SNIb",
                "SNIc",
                "SNIbc",
                "SNIc-BL",
                "21",
                "20",
                "27",
                "26",
                "25",
            ],
            "SN II": ["SN IIP", "SN IIL", "SNII", "SNIIP", "32", "30", "31"],
            "SN IIn": ["SNIIn", "35", "SLSN-II", "SLSNII"],
            "SLSN-I": ["40", "SLSN"],
            "SN Ibn": ["SN Ibn", "SNIbn", "SNIcn", "SN Icn", "SN Ibn/Icn"],
            "TDE": ["TDE", "42", "TDE-H-He", "TDE-He"]
        }

    @classmethod
    def canonicalize(cls, leaves, mapping, label):
        """Returns a canonical label, using the proper and alternative namings for
        each supernova class.

        Parameters
        ----------
        cls : SupernovaClass
            The SupernovaClass class.
        leaves : list
            The list of valid Supernova classifications
        label : str
            The label to canonicalize

        Returns
        -------
        str
            original label if already canonical, supernova class string if found in
            dictionary of alternative names, or the original label if not able to be
            canonicalized.
        """
        #print(f"label: {label}\nmapping: {mapping}\n")
        if label in leaves:
            return label
        elif mapping != None and label in mapping.keys():
            return mapping[label]
        
        alts = cls.get_alternative_namings()
        for canon_label, other_names in alts.items():
            if label in other_names:
                if label in mapping.keys():
                    return mapping[canon_label]
                return label
        return label

    

    
