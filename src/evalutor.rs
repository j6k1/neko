use usiagent::shogi::{Banmen, MochigomaCollections, Teban};

const PIECE_SCORE_MAP:[i32; 29] = [
    100,
    500,
    600,
    800,
    900,
    1300,
    1500,
    9999,
    1100,
    1000,
    1000,
    900,
    1500,
    1700,
    -100,
    -500,
    -600,
    -800,
    -900,
    -1300,
    -1500,
    -9999,
    -1100,
    -1000,
    -1000,
    -900,
    -1500,
    -1700,
    0
];

const HAND_SCORE_MAP: [i32; 7] = [
    120,550,660,880,990,1400,1650
];
pub struct Evalutor {

}

impl Evalutor {
    pub fn new() -> Evalutor {
        Evalutor {}
    }

    pub fn evalute(&self,teban:Teban,banmen:&Banmen,mc:&MochigomaCollections) -> i32 {
        let mut score = 0;

        for y in 0..9 {
            for x in 0..9 {
                let (x,y) = if teban == Teban::Sente {
                    (x,y)
                } else {
                    (8-x,8-y)
                };

                score += PIECE_SCORE_MAP[banmen.0[y][x] as usize];
            }
        }

        match mc {
            &MochigomaCollections::Pair(ref mc,_) if teban == Teban::Sente => {
                for (m,c) in mc.iter() {
                    score += HAND_SCORE_MAP[m as usize] * c as i32;
                }
            },
            &MochigomaCollections::Pair(_, ref mc) => {
                for (m,c) in mc.iter() {
                    score += HAND_SCORE_MAP[m as usize] * c as i32;
                }
            },
            _ => ()
        }

        score
    }
}