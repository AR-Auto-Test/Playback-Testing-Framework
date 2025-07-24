package androidx.activity;

import android.annotation.SuppressLint;
import b.a.b;
import b.t.e;
import b.t.f;
import b.t.h;
import b.t.i;
import java.util.ArrayDeque;
import java.util.Iterator;

/* loaded from: classes.dex */
public final class OnBackPressedDispatcher {

    /* renamed from: a  reason: collision with root package name */
    public final Runnable f51a;

    /* renamed from: b  reason: collision with root package name */
    public final ArrayDeque<b> f52b = new ArrayDeque<>();

    /* loaded from: classes.dex */
    public class LifecycleOnBackPressedCancellable implements f, b.a.a {

        /* renamed from: a  reason: collision with root package name */
        public final e f53a;

        /* renamed from: b  reason: collision with root package name */
        public final b f54b;

        /* renamed from: c  reason: collision with root package name */
        public b.a.a f55c;

        public LifecycleOnBackPressedCancellable(e eVar, b bVar) {
            this.f53a = eVar;
            this.f54b = bVar;
            eVar.a(this);
        }

        @Override // b.a.a
        public void cancel() {
            ((i) this.f53a).f2578a.e(this);
            this.f54b.f531b.remove(this);
            b.a.a aVar = this.f55c;
            if (aVar != null) {
                aVar.cancel();
                this.f55c = null;
            }
        }

        @Override // b.t.f
        public void e(h hVar, e.a aVar) {
            if (aVar == e.a.ON_START) {
                OnBackPressedDispatcher onBackPressedDispatcher = OnBackPressedDispatcher.this;
                b bVar = this.f54b;
                onBackPressedDispatcher.f52b.add(bVar);
                a aVar2 = new a(bVar);
                bVar.f531b.add(aVar2);
                this.f55c = aVar2;
            } else if (aVar == e.a.ON_STOP) {
                b.a.a aVar3 = this.f55c;
                if (aVar3 != null) {
                    aVar3.cancel();
                }
            } else if (aVar == e.a.ON_DESTROY) {
                cancel();
            }
        }
    }

    /* loaded from: classes.dex */
    public class a implements b.a.a {

        /* renamed from: a  reason: collision with root package name */
        public final b f57a;

        public a(b bVar) {
            this.f57a = bVar;
        }

        @Override // b.a.a
        public void cancel() {
            OnBackPressedDispatcher.this.f52b.remove(this.f57a);
            this.f57a.f531b.remove(this);
        }
    }

    public OnBackPressedDispatcher(Runnable runnable) {
        this.f51a = runnable;
    }

    @SuppressLint({"LambdaLast"})
    public void a(h hVar, b bVar) {
        e lifecycle = hVar.getLifecycle();
        if (((i) lifecycle).f2579b == e.b.DESTROYED) {
            return;
        }
        bVar.f531b.add(new LifecycleOnBackPressedCancellable(lifecycle, bVar));
    }

    public void b() {
        Iterator<b> descendingIterator = this.f52b.descendingIterator();
        while (descendingIterator.hasNext()) {
            b next = descendingIterator.next();
            if (next.f530a) {
                next.a();
                return;
            }
        }
        Runnable runnable = this.f51a;
        if (runnable != null) {
            runnable.run();
        }
    }
}