package b.w.b;

import android.view.View;
import androidx.recyclerview.widget.RecyclerView;

/* compiled from: SnapHelper.java */
/* loaded from: classes.dex */
public abstract class x extends RecyclerView.r {

    /* renamed from: a  reason: collision with root package name */
    public RecyclerView f2803a;

    /* renamed from: b  reason: collision with root package name */
    public final RecyclerView.t f2804b = new a();

    /* compiled from: SnapHelper.java */
    /* loaded from: classes.dex */
    public class a extends RecyclerView.t {

        /* renamed from: a  reason: collision with root package name */
        public boolean f2805a = false;

        public a() {
        }

        @Override // androidx.recyclerview.widget.RecyclerView.t
        public void onScrollStateChanged(RecyclerView recyclerView, int i) {
            super.onScrollStateChanged(recyclerView, i);
            if (i == 0 && this.f2805a) {
                this.f2805a = false;
                x.this.c();
            }
        }

        @Override // androidx.recyclerview.widget.RecyclerView.t
        public void onScrolled(RecyclerView recyclerView, int i, int i2) {
            if (i == 0 && i2 == 0) {
                return;
            }
            this.f2805a = true;
        }
    }

    public abstract int[] a(RecyclerView.o oVar, View view);

    public abstract View b(RecyclerView.o oVar);

    public void c() {
        RecyclerView.o layoutManager;
        View b2;
        RecyclerView recyclerView = this.f2803a;
        if (recyclerView == null || (layoutManager = recyclerView.getLayoutManager()) == null || (b2 = b(layoutManager)) == null) {
            return;
        }
        int[] a2 = a(layoutManager, b2);
        if (a2[0] == 0 && a2[1] == 0) {
            return;
        }
        this.f2803a.smoothScrollBy(a2[0], a2[1]);
    }
}