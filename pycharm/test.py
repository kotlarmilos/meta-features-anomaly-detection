from secrets import token_bytes
from coincurve import PublicKey
from sha3 import sha3_256


private_key = sha3_256(token_bytes(32)).digest()
public_key = PublicKey.from_valid_secret(private_key).format(compressed=False)[1:]
addr = sha3_256(public_key).digest()[-20:]

print('private_key:', private_key.hex())
print('eth addr: 0x' + addr.hex())