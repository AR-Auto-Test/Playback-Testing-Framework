package b.w.b;

import android.annotation.SuppressLint;
import android.os.Trace;
import androidx.recyclerview.widget.RecyclerView;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.concurrent.TimeUnit;

/* compiled from: GapWorker.java */
/* loaded from: classes.dex */
public final class m implements Runnable {

    /* renamed from: b  reason: collision with root package name */
    public static final ThreadLocal<m> f2770b = new ThreadLocal<>();

    /* renamed from: c  reason: collision with root package name */
    public static Comparator<c> f2771c = new a();

    /* renamed from: e  reason: collision with root package name */
    public long f2773e;

    /* renamed from: f  reason: collision with root package name */
    public long f2774f;

    /* renamed from: d  reason: collision with root package name */
    public ArrayList<RecyclerView> f2772d = new ArrayList<>();

    /* renamed from: g  reason: collision with root package name */
    public ArrayList<c> f2775g = new ArrayList<>();

    /* compiled from: GapWorker.java */
    /* loaded from: classes.dex */
    public static class a implements Comparator<c> {
        /* JADX DEBUG: Method arguments types fixed to match base method, original types: [java.lang.Object, java.lang.Object] */
        /* JADX WARN: Code restructure failed: missing block: B:12:0x0017, code lost:
            if (r0 == null) goto L10;
         */
        /* JADX WARN: Code restructure failed: missing block: B:17:0x0023, code lost:
            if (r0 != false) goto L13;
         */
        /* JADX WARN: Code restructure failed: missing block: B:26:?, code lost:
            return 1;
         */
        /* JADX WARN: Code restructure failed: missing block: B:27:?, code lost:
            return -1;
         */
        @Override // java.util.Comparator
        /*
            Code decompiled incorrectly, please refer to instructions dump.
        */
        public int compare(c cVar, c cVar2) {
            c cVar3 = cVar;
            c cVar4 = cVar2;
            RecyclerView recyclerView = cVar3.f2783d;
            if ((recyclerView == null) == (cVar4.f2783d == null)) {
                boolean z = cVar3.f2780a;
                if (z == cVar4.f2780a) {
                    int i = cVar4.f2781b - cVar3.f2781b;
                    if (i != 0) {
                        return i;
                    }
                    int i2 = cVar3.f2782c - cVar4.f2782c;
                    if (i2 != 0) {
                        return i2;
                    }
                    return 0;
                }
            }
        }
    }

    /* compiled from: GapWorker.java */
    @SuppressLint({"VisibleForTests"})
    /* loaded from: classes.dex */
    public static class b implements RecyclerView.o.c {

        /* renamed from: a  reason: collision with root package name */
        public int f2776a;

        /* renamed from: b  reason: collision with root package name */
        public int f2777b;

        /* renamed from: c  reason: collision with root package name */
        public int[] f2778c;

        /* renamed from: d  reason: collision with root package name */
        public int f2779d;

        public void a(int i, int i2) {
            if (i < 0) {
                throw new IllegalArgumentException("Layout positions must be non-negative");
            }
            if (i2 >= 0) {
                int i3 = this.f2779d * 2;
                int[] iArr = this.f2778c;
                if (iArr == null) {
                    int[] iArr2 = new int[4];
                    this.f2778c = iArr2;
                    Arrays.fill(iArr2, -1);
                } else if (i3 >= iArr.length) {
                    int[] iArr3 = new int[i3 * 2];
                    this.f2778c = iArr3;
                    System.arraycopy(iArr, 0, iArr3, 0, iArr.length);
                }
                int[] iArr4 = this.f2778c;
                iArr4[i3] = i;
                iArr4[i3 + 1] = i2;
                this.f2779d++;
                return;
            }
            throw new IllegalArgumentException("Pixel distance must be non-negative");
        }

        public void b(RecyclerView recyclerView, boolean z) {
            this.f2779d = 0;
            int[] iArr = this.f2778c;
            if (iArr != null) {
                Arrays.fill(iArr, -1);
            }
            RecyclerView.o oVar = recyclerView.mLayout;
            if (recyclerView.mAdapter == null || oVar == null || !oVar.isItemPrefetchEnabled()) {
                return;
            }
            if (z) {
                if (!recyclerView.mAdapterHelper.g()) {
                    oVar.collectInitialPrefetchPositions(recyclerView.mAdapter.getItemCount(), this);
                }
            } else if (!recyclerView.hasPendingAdapterUpdates()) {
                oVar.collectAdjacentPrefetchPositions(this.f2776a, this.f2777b, recyclerView.mState, this);
            }
            int i = this.f2779d;
            if (i > oVar.mPrefetchMaxCountObserved) {
                oVar.mPrefetchMaxCountObserved = i;
                oVar.mPrefetchMaxObservedInInitialPrefetch = z;
                recyclerView.mRecycler.m();
            }
        }

        public boolean c(int i) {
            if (this.f2778c != null) {
                int i2 = this.f2779d * 2;
                for (int i3 = 0; i3 < i2; i3 += 2) {
                    if (this.f2778c[i3] == i) {
                        return true;
                    }
                }
            }
            return false;
        }
    }

    /* compiled from: GapWorker.java */
    /* loaded from: classes.dex */
    public static class c {

        /* renamed from: a  reason: collision with root package name */
        public boolean f2780a;

        /* renamed from: b  reason: collision with root package name */
        public int f2781b;

        /* renamed from: c  reason: collision with root package name */
        public int f2782c;

        /* renamed from: d  reason: collision with root package name */
        public RecyclerView f2783d;

        /* renamed from: e  reason: collision with root package name */
        public int f2784e;
    }

    public void a(RecyclerView recyclerView, int i, int i2) {
        if (recyclerView.isAttachedToWindow() && this.f2773e == 0) {
            this.f2773e = recyclerView.getNanoTime();
            recyclerView.post(this);
        }
        b bVar = recyclerView.mPrefetchRegistry;
        bVar.f2776a = i;
        bVar.f2777b = i2;
    }

    public void b(long j) {
        c cVar;
        RecyclerView recyclerView;
        RecyclerView recyclerView2;
        c cVar2;
        int size = this.f2772d.size();
        int i = 0;
        for (int i2 = 0; i2 < size; i2++) {
            RecyclerView recyclerView3 = this.f2772d.get(i2);
            if (recyclerView3.getWindowVisibility() == 0) {
                recyclerView3.mPrefetchRegistry.b(recyclerView3, false);
                i += recyclerView3.mPrefetchRegistry.f2779d;
            }
        }
        this.f2775g.ensureCapacity(i);
        int i3 = 0;
        for (int i4 = 0; i4 < size; i4++) {
            RecyclerView recyclerView4 = this.f2772d.get(i4);
            if (recyclerView4.getWindowVisibility() == 0) {
                b bVar = recyclerView4.mPrefetchRegistry;
                int abs = Math.abs(bVar.f2777b) + Math.abs(bVar.f2776a);
                for (int i5 = 0; i5 < bVar.f2779d * 2; i5 += 2) {
                    if (i3 >= this.f2775g.size()) {
                        cVar2 = new c();
                        this.f2775g.add(cVar2);
                    } else {
                        cVar2 = this.f2775g.get(i3);
                    }
                    int[] iArr = bVar.f2778c;
                    int i6 = iArr[i5 + 1];
                    cVar2.f2780a = i6 <= abs;
                    cVar2.f2781b = abs;
                    cVar2.f2782c = i6;
                    cVar2.f2783d = recyclerView4;
                    cVar2.f2784e = iArr[i5];
                    i3++;
                }
            }
        }
        Collections.sort(this.f2775g, f2771c);
        for (int i7 = 0; i7 < this.f2775g.size() && (recyclerView = (cVar = this.f2775g.get(i7)).f2783d) != null; i7++) {
            RecyclerView.d0 c2 = c(recyclerView, cVar.f2784e, cVar.f2780a ? RecyclerView.FOREVER_NS : j);
            if (c2 != null && c2.mNestedRecyclerView != null && c2.isBound() && !c2.isInvalid() && (recyclerView2 = c2.mNestedRecyclerView.get()) != null) {
                if (recyclerView2.mDataSetHasChangedAfterLayout && recyclerView2.mChildHelper.h() != 0) {
                    recyclerView2.removeAndRecycleViews();
                }
                b bVar2 = recyclerView2.mPrefetchRegistry;
                bVar2.b(recyclerView2, true);
                if (bVar2.f2779d != 0) {
                    try {
                        int i8 = b.j.f.e.f2121a;
                        Trace.beginSection(RecyclerView.TRACE_NESTED_PREFETCH_TAG);
                        RecyclerView.a0 a0Var = recyclerView2.mState;
                        RecyclerView.g gVar = recyclerView2.mAdapter;
                        a0Var.f393d = 1;
                        a0Var.f394e = gVar.getItemCount();
                        a0Var.f396g = false;
                        a0Var.f397h = false;
                        a0Var.i = false;
                        for (int i9 = 0; i9 < bVar2.f2779d * 2; i9 += 2) {
                            c(recyclerView2, bVar2.f2778c[i9], j);
                        }
                        Trace.endSection();
                    } catch (Throwable th) {
                        int i10 = b.j.f.e.f2121a;
                        Trace.endSection();
                        throw th;
                    }
                } else {
                    continue;
                }
            }
            cVar.f2780a = false;
            cVar.f2781b = 0;
            cVar.f2782c = 0;
            cVar.f2783d = null;
            cVar.f2784e = 0;
        }
    }

    public final RecyclerView.d0 c(RecyclerView recyclerView, int i, long j) {
        boolean z;
        int h2 = recyclerView.mChildHelper.h();
        int i2 = 0;
        while (true) {
            if (i2 >= h2) {
                z = false;
                break;
            }
            RecyclerView.d0 childViewHolderInt = RecyclerView.getChildViewHolderInt(recyclerView.mChildHelper.g(i2));
            if (childViewHolderInt.mPosition == i && !childViewHolderInt.isInvalid()) {
                z = true;
                break;
            }
            i2++;
        }
        if (z) {
            return null;
        }
        RecyclerView.v vVar = recyclerView.mRecycler;
        try {
            recyclerView.onEnterLayoutOrScroll();
            RecyclerView.d0 k = vVar.k(i, false, j);
            if (k != null) {
                if (k.isBound() && !k.isInvalid()) {
                    vVar.h(k.itemView);
                } else {
                    vVar.a(k, false);
                }
            }
            return k;
        } finally {
            recyclerView.onExitLayoutOrScroll(false);
        }
    }

    @Override // java.lang.Runnable
    public void run() {
        try {
            int i = b.j.f.e.f2121a;
            Trace.beginSection(RecyclerView.TRACE_PREFETCH_TAG);
            if (this.f2772d.isEmpty()) {
                this.f2773e = 0L;
                Trace.endSection();
                return;
            }
            int size = this.f2772d.size();
            long j = 0;
            for (int i2 = 0; i2 < size; i2++) {
                RecyclerView recyclerView = this.f2772d.get(i2);
                if (recyclerView.getWindowVisibility() == 0) {
                    j = Math.max(recyclerView.getDrawingTime(), j);
                }
            }
            if (j == 0) {
                this.f2773e = 0L;
                Trace.endSection();
                return;
            }
            b(TimeUnit.MILLISECONDS.toNanos(j) + this.f2774f);
            this.f2773e = 0L;
            Trace.endSection();
        } catch (Throwable th) {
            this.f2773e = 0L;
            int i3 = b.j.f.e.f2121a;
            Trace.endSection();
            throw th;
        }
    }
}