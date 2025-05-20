from .maker import Maker
from controllers.mac import MAC


class MACMaker(Maker):
    """Factory class for creating Controllers."""

    @staticmethod
    def make_basic_mac(*args, **kwargs) -> MAC:
        from controllers.basic_controller import BasicMAC
        return BasicMAC(*args, **kwargs)

    @staticmethod
    def make_non_shared_mac(*args, **kwargs) -> MAC:
        from controllers.non_shared_controller import NonSharedMAC
        return NonSharedMAC(*args, **kwargs)    # TODO migrate NonSharedMAC to MAC

    @staticmethod
    def make_maddpg_mac(*args, **kwargs) -> MAC:
        from controllers.maddpg_controller import MADDPGMAC
        return MADDPGMAC(*args, **kwargs)   # TODO migrate NonSharedMAC to MAC

    @staticmethod
    def make_entity_mac(*args, **kwargs) -> MAC:
        from controllers.entity_controller import EntityMAC
        return EntityMAC(*args, **kwargs)

    @staticmethod
    def make_pymarl2_nmac_controller(*args, **kwargs) -> MAC:
        """MAC Implementation from PyMARL2"""
        from controllers.pymarl2_nmac_controller import NMAC
        return NMAC(*args, **kwargs)

    @staticmethod
    def make_entity_maic_mac(*args, **kwargs) -> MAC:
        from controllers.entity_maic_controller import EntityMAICMAC
        return EntityMAICMAC(*args, **kwargs)

    def make_imagine_mac(*args, **kwargs) -> MAC:
        from controllers.imagine_controller import ImagineMAC
        return ImagineMAC(*args, **kwargs)

    @staticmethod
    def make_icm_mac(*args, **kwargs) -> MAC:
        """MAC Implementation from PyMARL2"""
        from controllers.icm_controller import ICMMAC
        return ICMMAC(*args, **kwargs)

    @staticmethod
    def make_entity_state_as_obs_mac(*args, **kwargs) -> MAC:
        from controllers.entity_state_as_obs_controller import EntityStateAsObsMAC
        return EntityStateAsObsMAC(*args, **kwargs)

    @staticmethod
    def make_pymarl2_nmac_controller_with_t(*args, **kwargs) -> "NMAC":
        """MAC Implementation from PyMARL2"""
        from controllers.pymarl2_nmac_controller_with_t import NMAC
        return NMAC(*args, **kwargs)

    @staticmethod
    def make_new_basic_mac(*args, **kwargs) -> MAC:
        """A more efficient BasicMAC along with new q learner."""
        from controllers.new_basic_controller import BasicMAC
        return BasicMAC(*args, **kwargs)

    @staticmethod
    def make_maic_mac(*args, **kwargs) -> MAC:
        """Entity agent using attention to merge obs."""
        from controllers.maic_controller import MAICMAC
        return MAICMAC(*args, **kwargs)

    @staticmethod
    def make_cacom_mac(*args, **kwargs) -> MAC:
        """Entity agent using attention to merge obs."""
        from controllers.cacom_controller import CACOM_MAC
        return CACOM_MAC(*args, **kwargs)

    @staticmethod
    def make_tmac_p2p_comm_mac(*args, **kwargs) -> MAC:
        """Entity agent using attention to merge obs."""
        from controllers.tmac_p2p_comm_controller import VffacMAC
        return VffacMAC(*args, **kwargs)

    @staticmethod
    def make_basic_mac_7(*args, **kwargs) -> MAC:
        """A more efficient BasicMAC along with new q learner."""
        from controllers.basic_controller_7 import BasicMAC_7
        return BasicMAC_7(*args, **kwargs)
