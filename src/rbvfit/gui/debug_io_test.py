#!/usr/bin/env python
"""
Simple debug test - minimal reproduction
"""

def test_rb_spectrum_direct():
    """Test rb_spectrum exactly as you showed"""
    from rbcodes.utils.rb_spectrum import rb_spectrum
    
    # Use the exact filename that works for you
    filename = '/Users/bordoloi/WORK/python/rbvfit/src/rbvfit/examples/J1148_18887_HIRES_OI1302.json'
    print(f"Loading: {filename}")
    
    sp = rb_spectrum.from_file(filename)
    print(f"sp object: {sp}")
    print(f"sp.wavelength: {type(sp.wavelength)}")
    print(f"sp.flux: {type(sp.flux)}")
    print(f"sp.sig: {type(sp.sig)}")
    
    # Try to access all possible error attributes
    print("\nChecking for error attributes:")
    for attr in dir(sp):
        if 'sig' in attr.lower() or 'err' in attr.lower() or 'unc' in attr.lower():
            val = getattr(sp, attr)
            print(f"  {attr}: {type(val)}")
    
    # Try the manual approach
    print("\nManual extraction:")
    wave = sp.wavelength.value
    flux = sp.flux.value
    
    print(f"wave: {wave.shape} {type(wave)}")
    print(f"flux: {flux.shape} {type(flux)}")
    
    # Try different error possibilities
    if sp.sig is not None:
        error = sp.sig.value
        print(f"error from sp.sig: {error.shape} {type(error)}")
    else:
        print("sp.sig is None - checking alternatives...")
        
        # Check for other error attributes
        if hasattr(sp, 'error') and sp.error is not None:
            error = sp.error.value
            print(f"error from sp.error: {error.shape} {type(error)}")
        elif hasattr(sp, 'uncertainty') and sp.uncertainty is not None:
            error = sp.uncertainty.value  
            print(f"error from sp.uncertainty: {error.shape} {type(error)}")
        else:
            print("No error array found - creating default")
            error = 0.05 * flux
            print(f"default error: {error.shape} {type(error)}")
    
    return wave, flux, error

if __name__ == "__main__":
    import os
    print(f"Current directory: {os.getcwd()}")
    print(f"Files in current directory: {os.listdir('.')}")
    
    try:
        wave, flux, error = test_rb_spectrum_direct()
        print(f"\nSUCCESS!")
        print(f"Final arrays: wave{wave.shape}, flux{flux.shape}, error{error.shape}")
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()