package androidx.lifecycle;

import b.t.c;
import b.t.e;
import b.t.f;
import b.t.h;

/* loaded from: classes.dex */
public class FullLifecycleObserverAdapter implements f {

    /* renamed from: a  reason: collision with root package name */
    public final c f309a;

    /* renamed from: b  reason: collision with root package name */
    public final f f310b;

    public FullLifecycleObserverAdapter(c cVar, f fVar) {
        this.f309a = cVar;
        this.f310b = fVar;
    }

    @Override // b.t.f
    public void e(h hVar, e.a aVar) {
        switch (aVar.ordinal()) {
            case 0:
                this.f309a.d(hVar);
                break;
            case 1:
                this.f309a.onStart(hVar);
                break;
            case 2:
                this.f309a.c(hVar);
                break;
            case 3:
                this.f309a.f(hVar);
                break;
            case 4:
                this.f309a.onStop(hVar);
                break;
            case 5:
                this.f309a.onDestroy(hVar);
                break;
            case 6:
                throw new IllegalArgumentException("ON_ANY must not been send by anybody");
        }
        f fVar = this.f310b;
        if (fVar != null) {
            fVar.e(hVar, aVar);
        }
    }
}