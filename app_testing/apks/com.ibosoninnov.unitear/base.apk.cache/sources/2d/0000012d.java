package b.a;

import java.util.Iterator;
import java.util.concurrent.CopyOnWriteArrayList;

/* compiled from: OnBackPressedCallback.java */
/* loaded from: classes.dex */
public abstract class b {

    /* renamed from: a  reason: collision with root package name */
    public boolean f530a;

    /* renamed from: b  reason: collision with root package name */
    public CopyOnWriteArrayList<a> f531b = new CopyOnWriteArrayList<>();

    public b(boolean z) {
        this.f530a = z;
    }

    public abstract void a();

    public final void b() {
        Iterator<a> it = this.f531b.iterator();
        while (it.hasNext()) {
            it.next().cancel();
        }
    }
}