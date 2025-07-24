package c.e.b.p000if;

import android.content.Context;
import android.view.GestureDetector;
import android.view.MotionEvent;
import android.view.View;
import androidx.recyclerview.widget.RecyclerView;

/* compiled from: RecyclerViewItemClickListener.java */
/* renamed from: c.e.b.if.o  reason: invalid package */
/* loaded from: classes2.dex */
public class o implements RecyclerView.s {

    /* renamed from: a  reason: collision with root package name */
    public GestureDetector f4895a;

    /* renamed from: b  reason: collision with root package name */
    public h f4896b;

    /* compiled from: RecyclerViewItemClickListener.java */
    /* renamed from: c.e.b.if.o$a */
    /* loaded from: classes2.dex */
    public class a extends GestureDetector.SimpleOnGestureListener {

        /* renamed from: a  reason: collision with root package name */
        public final /* synthetic */ RecyclerView f4897a;

        /* renamed from: b  reason: collision with root package name */
        public final /* synthetic */ h f4898b;

        public a(o oVar, RecyclerView recyclerView, h hVar) {
            this.f4897a = recyclerView;
            this.f4898b = hVar;
        }

        @Override // android.view.GestureDetector.SimpleOnGestureListener, android.view.GestureDetector.OnGestureListener
        public void onLongPress(MotionEvent motionEvent) {
            h hVar;
            View findChildViewUnder = this.f4897a.findChildViewUnder(motionEvent.getX(), motionEvent.getY());
            if (findChildViewUnder == null || (hVar = this.f4898b) == null) {
                return;
            }
            hVar.b(findChildViewUnder, this.f4897a.getChildLayoutPosition(findChildViewUnder));
        }

        @Override // android.view.GestureDetector.SimpleOnGestureListener, android.view.GestureDetector.OnGestureListener
        public boolean onSingleTapUp(MotionEvent motionEvent) {
            return true;
        }
    }

    public o(Context context, RecyclerView recyclerView, h hVar) {
        this.f4896b = hVar;
        this.f4895a = new GestureDetector(context, new a(this, recyclerView, hVar));
    }

    @Override // androidx.recyclerview.widget.RecyclerView.s
    public void a(RecyclerView recyclerView, MotionEvent motionEvent) {
    }

    @Override // androidx.recyclerview.widget.RecyclerView.s
    public boolean b(RecyclerView recyclerView, MotionEvent motionEvent) {
        View findChildViewUnder = recyclerView.findChildViewUnder(motionEvent.getX(), motionEvent.getY());
        if (findChildViewUnder == null || this.f4896b == null || !this.f4895a.onTouchEvent(motionEvent)) {
            return false;
        }
        this.f4896b.a(findChildViewUnder, recyclerView.getChildLayoutPosition(findChildViewUnder));
        return false;
    }

    @Override // androidx.recyclerview.widget.RecyclerView.s
    public void c(boolean z) {
    }
}