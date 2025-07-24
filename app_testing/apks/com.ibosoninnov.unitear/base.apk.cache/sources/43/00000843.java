package c.c.a.m.x.g;

import android.util.Log;
import c.c.a.m.p;
import c.c.a.m.s;
import c.c.a.m.v.w;
import java.io.File;
import java.io.IOException;

/* compiled from: GifDrawableEncoder.java */
/* loaded from: classes.dex */
public class d implements s<c> {
    @Override // c.c.a.m.d
    public boolean a(Object obj, File file, p pVar) {
        try {
            c.c.a.s.a.b(((c) ((w) obj).get()).f4036b.f4043a.f4045a.e().asReadOnlyBuffer(), file);
            return true;
        } catch (IOException e2) {
            if (Log.isLoggable("GifEncoder", 5)) {
                Log.w("GifEncoder", "Failed to encode GIF drawable data", e2);
            }
            return false;
        }
    }

    @Override // c.c.a.m.s
    public c.c.a.m.c b(p pVar) {
        return c.c.a.m.c.SOURCE;
    }
}