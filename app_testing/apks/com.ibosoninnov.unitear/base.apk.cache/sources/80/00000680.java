package c.a.a.x.b;

import android.graphics.Path;
import android.graphics.PathMeasure;
import java.util.ArrayList;
import java.util.List;

/* compiled from: CompoundTrimPathContent.java */
/* loaded from: classes.dex */
public class b {

    /* renamed from: a  reason: collision with root package name */
    public List<s> f3149a = new ArrayList();

    public void a(Path path) {
        for (int size = this.f3149a.size() - 1; size >= 0; size--) {
            s sVar = this.f3149a.get(size);
            PathMeasure pathMeasure = c.a.a.c0.g.f3031a;
            if (sVar != null && !sVar.f3217a) {
                c.a.a.c0.g.a(path, ((c.a.a.x.c.c) sVar.f3220d).j() / 100.0f, ((c.a.a.x.c.c) sVar.f3221e).j() / 100.0f, ((c.a.a.x.c.c) sVar.f3222f).j() / 360.0f);
            }
        }
    }
}