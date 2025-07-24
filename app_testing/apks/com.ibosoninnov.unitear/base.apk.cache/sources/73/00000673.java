package c.a.a;

import java.util.ArrayList;
import java.util.Iterator;

/* compiled from: LottieTask.java */
/* loaded from: classes.dex */
public class q implements Runnable {

    /* renamed from: b  reason: collision with root package name */
    public final /* synthetic */ r f3124b;

    public q(r rVar) {
        this.f3124b = rVar;
    }

    /* JADX DEBUG: Multi-variable search result rejected for r1v0, resolved type: V */
    /* JADX WARN: Multi-variable type inference failed */
    @Override // java.lang.Runnable
    public void run() {
        if (this.f3124b.f3129e == null) {
            return;
        }
        p<T> pVar = this.f3124b.f3129e;
        V v = pVar.f3122a;
        if (v != 0) {
            r rVar = this.f3124b;
            synchronized (rVar) {
                Iterator it = new ArrayList(rVar.f3126b).iterator();
                while (it.hasNext()) {
                    ((l) it.next()).a(v);
                }
            }
            return;
        }
        r rVar2 = this.f3124b;
        Throwable th = pVar.f3123b;
        synchronized (rVar2) {
            ArrayList arrayList = new ArrayList(rVar2.f3127c);
            if (arrayList.isEmpty()) {
                c.a.a.c0.c.c("Lottie encountered an error but no failure listener was added:", th);
                return;
            }
            Iterator it2 = arrayList.iterator();
            while (it2.hasNext()) {
                ((l) it2.next()).a(th);
            }
        }
    }
}