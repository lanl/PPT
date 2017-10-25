class SimianError(Exception):
  def __init__(self, value): self.value = str(value)

  def __str__(self): return self.value
