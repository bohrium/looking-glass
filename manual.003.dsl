(
split<int><gridpair> (many noise) \side:int -> (
split<int><gridpair> (afew noise) \nb_objs:int -> (
split<grid><gridpair> (new_grid side side) \blank:grid -> (
split<color><gridpair> (rainbow noise) \color_a:grid -> (
split<color><gridpair> (rainbow noise) \color_b:grid -> (
repeat<gridpair> nb_objs (pair<gridpair> blank blank) \xy:gridpair -> (
    split<ptdgrid><gridpair>
    (reserve_shape noise (snd<gridpair> xy) large_square (center large_square)) \pg:ptdgrid -> (
    pair<gridpair> (
        split<grid><grid>
        (paint_sprite (fst<gridpair> xy) (monochrome small_plus color_a) (snd<gridpair> pg) (center small_plus)) \half_painted:grid -> (
        paint_cell half_painted (snd<ptdgrid> pg) color_b
        )
    )(
        split<grid><grid>
        (paint_sprite (fst<ptdgrid> pg) (monochrome large_plus color_a) (snd<ptdgrid> pg) (center large_plus)) \half_painted:grid -> (
        paint_sprite half_painted (monochrome large_times color_b) (snd<ptdgrid> pg) (center large_times)
        )
    ))
))))))
)
