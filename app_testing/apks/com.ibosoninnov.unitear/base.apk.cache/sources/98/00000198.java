package b.b.h;

import android.content.Context;
import android.content.res.Configuration;
import android.content.res.TypedArray;
import android.util.AttributeSet;
import android.util.TypedValue;
import android.view.ContextThemeWrapper;
import android.view.MotionEvent;
import android.view.View;
import android.view.ViewGroup;
import androidx.appcompat.widget.ActionMenuView;
import com.google.android.material.internal.StaticLayoutBuilderCompat;
import com.ibosoninnov.unitear.R;

/* compiled from: AbsActionBarView.java */
/* loaded from: classes.dex */
public abstract class a extends ViewGroup {

    /* renamed from: b  reason: collision with root package name */
    public final C0009a f769b;

    /* renamed from: c  reason: collision with root package name */
    public final Context f770c;

    /* renamed from: d  reason: collision with root package name */
    public ActionMenuView f771d;

    /* renamed from: e  reason: collision with root package name */
    public c f772e;

    /* renamed from: f  reason: collision with root package name */
    public int f773f;

    /* renamed from: g  reason: collision with root package name */
    public b.j.j.s f774g;

    /* renamed from: h  reason: collision with root package name */
    public boolean f775h;
    public boolean i;

    /* compiled from: AbsActionBarView.java */
    /* renamed from: b.b.h.a$a  reason: collision with other inner class name */
    /* loaded from: classes.dex */
    public class C0009a implements b.j.j.t {

        /* renamed from: a  reason: collision with root package name */
        public boolean f776a = false;

        /* renamed from: b  reason: collision with root package name */
        public int f777b;

        public C0009a() {
        }

        @Override // b.j.j.t
        public void a(View view) {
            this.f776a = true;
        }

        @Override // b.j.j.t
        public void b(View view) {
            if (this.f776a) {
                return;
            }
            a aVar = a.this;
            aVar.f774g = null;
            a.super.setVisibility(this.f777b);
        }

        @Override // b.j.j.t
        public void c(View view) {
            a.super.setVisibility(0);
            this.f776a = false;
        }
    }

    public a(Context context, AttributeSet attributeSet) {
        this(context, attributeSet, 0);
    }

    public int c(View view, int i, int i2, int i3) {
        view.measure(View.MeasureSpec.makeMeasureSpec(i, Integer.MIN_VALUE), i2);
        return Math.max(0, (i - view.getMeasuredWidth()) - i3);
    }

    public int d(View view, int i, int i2, int i3, boolean z) {
        int measuredWidth = view.getMeasuredWidth();
        int measuredHeight = view.getMeasuredHeight();
        int i4 = ((i3 - measuredHeight) / 2) + i2;
        if (z) {
            view.layout(i - measuredWidth, i4, i, measuredHeight + i4);
        } else {
            view.layout(i, i4, i + measuredWidth, measuredHeight + i4);
        }
        return z ? -measuredWidth : measuredWidth;
    }

    public b.j.j.s e(int i, long j) {
        b.j.j.s sVar = this.f774g;
        if (sVar != null) {
            sVar.b();
        }
        if (i == 0) {
            if (getVisibility() != 0) {
                setAlpha(StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD);
            }
            b.j.j.s b2 = b.j.j.q.b(this);
            b2.a(1.0f);
            b2.c(j);
            C0009a c0009a = this.f769b;
            a.this.f774g = b2;
            c0009a.f777b = i;
            View view = b2.f2231a.get();
            if (view != null) {
                b2.e(view, c0009a);
            }
            return b2;
        }
        b.j.j.s b3 = b.j.j.q.b(this);
        b3.a(StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD);
        b3.c(j);
        C0009a c0009a2 = this.f769b;
        a.this.f774g = b3;
        c0009a2.f777b = i;
        View view2 = b3.f2231a.get();
        if (view2 != null) {
            b3.e(view2, c0009a2);
        }
        return b3;
    }

    public int getAnimatedVisibility() {
        if (this.f774g != null) {
            return this.f769b.f777b;
        }
        return getVisibility();
    }

    public int getContentHeight() {
        return this.f773f;
    }

    @Override // android.view.View
    public void onConfigurationChanged(Configuration configuration) {
        super.onConfigurationChanged(configuration);
        TypedArray obtainStyledAttributes = getContext().obtainStyledAttributes(null, b.b.b.f541a, R.attr.actionBarStyle, 0);
        setContentHeight(obtainStyledAttributes.getLayoutDimension(13, 0));
        obtainStyledAttributes.recycle();
        c cVar = this.f772e;
        if (cVar != null) {
            Configuration configuration2 = cVar.f688c.getResources().getConfiguration();
            int i = configuration2.screenWidthDp;
            int i2 = configuration2.screenHeightDp;
            cVar.r = (configuration2.smallestScreenWidthDp > 600 || i > 600 || (i > 960 && i2 > 720) || (i > 720 && i2 > 960)) ? 5 : (i >= 500 || (i > 640 && i2 > 480) || (i > 480 && i2 > 640)) ? 4 : i >= 360 ? 3 : 2;
            b.b.g.i.g gVar = cVar.f689d;
            if (gVar != null) {
                gVar.onItemsChanged(true);
            }
        }
    }

    @Override // android.view.View
    public boolean onHoverEvent(MotionEvent motionEvent) {
        int actionMasked = motionEvent.getActionMasked();
        if (actionMasked == 9) {
            this.i = false;
        }
        if (!this.i) {
            boolean onHoverEvent = super.onHoverEvent(motionEvent);
            if (actionMasked == 9 && !onHoverEvent) {
                this.i = true;
            }
        }
        if (actionMasked == 10 || actionMasked == 3) {
            this.i = false;
        }
        return true;
    }

    @Override // android.view.View
    public boolean onTouchEvent(MotionEvent motionEvent) {
        int actionMasked = motionEvent.getActionMasked();
        if (actionMasked == 0) {
            this.f775h = false;
        }
        if (!this.f775h) {
            boolean onTouchEvent = super.onTouchEvent(motionEvent);
            if (actionMasked == 0 && !onTouchEvent) {
                this.f775h = true;
            }
        }
        if (actionMasked == 1 || actionMasked == 3) {
            this.f775h = false;
        }
        return true;
    }

    public abstract void setContentHeight(int i);

    @Override // android.view.View
    public void setVisibility(int i) {
        if (i != getVisibility()) {
            b.j.j.s sVar = this.f774g;
            if (sVar != null) {
                sVar.b();
            }
            super.setVisibility(i);
        }
    }

    public a(Context context, AttributeSet attributeSet, int i) {
        super(context, attributeSet, i);
        this.f769b = new C0009a();
        TypedValue typedValue = new TypedValue();
        if (context.getTheme().resolveAttribute(R.attr.actionBarPopupTheme, typedValue, true) && typedValue.resourceId != 0) {
            this.f770c = new ContextThemeWrapper(context, typedValue.resourceId);
        } else {
            this.f770c = context;
        }
    }
}