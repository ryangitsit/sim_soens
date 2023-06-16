
function clear_all(strct)
    strct = nothing
    return strct
end

function unbindvariables()
    for name in names(Main)
        if !isconst(Main, name)
            Main.eval(:($name = nothing))
        end
    end
end