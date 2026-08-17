pub struct RegFile<'a> {
    pub regs: &'a mut [i16; 16],
}

impl<'a> RegFile<'a> {
    pub fn new(regs: &'a mut [i16; 16]) -> Self {
        Self { regs }
    }

    pub fn read(&self, idx: usize) -> i16 {
        self.regs[idx]
    }

    pub fn write(&mut self, idx: usize, value: i16) {
        self.regs[idx] = value;
    }
}
