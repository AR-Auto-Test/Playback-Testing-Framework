package androidx.appcompat.view.menu;

import android.content.Context;
import android.content.res.Configuration;
import android.content.res.Resources;
import android.content.res.TypedArray;
import android.graphics.drawable.Drawable;
import android.os.Parcelable;
import android.text.TextUtils;
import android.util.AttributeSet;
import android.view.MotionEvent;
import android.view.View;
import androidx.appcompat.widget.ActionMenuView;
import b.b.g.i.g;
import b.b.g.i.i;
import b.b.g.i.n;
import b.b.g.i.p;
import b.b.h.c;
import b.b.h.h0;
import b.b.h.z;

/* loaded from: classes.dex */
public class ActionMenuItemView extends z implements n.a, View.OnClickListener, ActionMenuView.a {

    /* renamed from: b  reason: collision with root package name */
    public i f79b;

    /* renamed from: c  reason: collision with root package name */
    public CharSequence f80c;

    /* renamed from: d  reason: collision with root package name */
    public Drawable f81d;

    /* renamed from: e  reason: collision with root package name */
    public g.b f82e;

    /* renamed from: f  reason: collision with root package name */
    public h0 f83f;

    /* renamed from: g  reason: collision with root package name */
    public b f84g;

    /* renamed from: h  reason: collision with root package name */
    public boolean f85h;
    public boolean i;
    public int j;
    public int k;
    public int l;

    /* loaded from: classes.dex */
    public class a extends h0 {
        public a() {
            super(ActionMenuItemView.this);
        }

        @Override // b.b.h.h0
        public p b() {
            c.a aVar;
            b bVar = ActionMenuItemView.this.f84g;
            if (bVar == null || (aVar = c.this.w) == null) {
                return null;
            }
            return aVar.a();
        }

        @Override // b.b.h.h0
        public boolean c() {
            p b2;
            ActionMenuItemView actionMenuItemView = ActionMenuItemView.this;
            g.b bVar = actionMenuItemView.f82e;
            return bVar != null && bVar.a(actionMenuItemView.f79b) && (b2 = b()) != null && b2.a();
        }
    }

    /* loaded from: classes.dex */
    public static abstract class b {
    }

    public ActionMenuItemView(Context context, AttributeSet attributeSet) {
        super(context, attributeSet, 0);
        Resources resources = context.getResources();
        this.f85h = d();
        TypedArray obtainStyledAttributes = context.obtainStyledAttributes(attributeSet, b.b.b.f543c, 0, 0);
        this.j = obtainStyledAttributes.getDimensionPixelSize(0, 0);
        obtainStyledAttributes.recycle();
        this.l = (int) ((resources.getDisplayMetrics().density * 32.0f) + 0.5f);
        setOnClickListener(this);
        this.k = -1;
        setSaveEnabled(false);
    }

    @Override // androidx.appcompat.widget.ActionMenuView.a
    public boolean a() {
        return c();
    }

    @Override // androidx.appcompat.widget.ActionMenuView.a
    public boolean b() {
        return c() && this.f79b.getIcon() == null;
    }

    public boolean c() {
        return !TextUtils.isEmpty(getText());
    }

    public final boolean d() {
        Configuration configuration = getContext().getResources().getConfiguration();
        int i = configuration.screenWidthDp;
        return i >= 480 || (i >= 640 && configuration.screenHeightDp >= 480) || configuration.orientation == 2;
    }

    public final void e() {
        boolean z = true;
        boolean z2 = !TextUtils.isEmpty(this.f80c);
        if (this.f81d != null) {
            if (!((this.f79b.y & 4) == 4) || (!this.f85h && !this.i)) {
                z = false;
            }
        }
        boolean z3 = z2 & z;
        setText(z3 ? this.f80c : null);
        CharSequence charSequence = this.f79b.q;
        if (TextUtils.isEmpty(charSequence)) {
            setContentDescription(z3 ? null : this.f79b.f734e);
        } else {
            setContentDescription(charSequence);
        }
        CharSequence charSequence2 = this.f79b.r;
        if (TextUtils.isEmpty(charSequence2)) {
            b.b.a.n(this, z3 ? null : this.f79b.f734e);
        } else {
            b.b.a.n(this, charSequence2);
        }
    }

    @Override // b.b.g.i.n.a
    public i getItemData() {
        return this.f79b;
    }

    @Override // b.b.g.i.n.a
    public void initialize(i iVar, int i) {
        this.f79b = iVar;
        setIcon(iVar.getIcon());
        setTitle(iVar.getTitleCondensed());
        setId(iVar.f730a);
        setVisibility(iVar.isVisible() ? 0 : 8);
        setEnabled(iVar.isEnabled());
        if (iVar.hasSubMenu() && this.f83f == null) {
            this.f83f = new a();
        }
    }

    @Override // android.view.View.OnClickListener
    public void onClick(View view) {
        g.b bVar = this.f82e;
        if (bVar != null) {
            bVar.a(this.f79b);
        }
    }

    @Override // android.widget.TextView, android.view.View
    public void onConfigurationChanged(Configuration configuration) {
        super.onConfigurationChanged(configuration);
        this.f85h = d();
        e();
    }

    @Override // b.b.h.z, android.widget.TextView, android.view.View
    public void onMeasure(int i, int i2) {
        int i3;
        int i4;
        boolean c2 = c();
        if (c2 && (i4 = this.k) >= 0) {
            super.setPadding(i4, getPaddingTop(), getPaddingRight(), getPaddingBottom());
        }
        super.onMeasure(i, i2);
        int mode = View.MeasureSpec.getMode(i);
        int size = View.MeasureSpec.getSize(i);
        int measuredWidth = getMeasuredWidth();
        if (mode == Integer.MIN_VALUE) {
            i3 = Math.min(size, this.j);
        } else {
            i3 = this.j;
        }
        if (mode != 1073741824 && this.j > 0 && measuredWidth < i3) {
            super.onMeasure(View.MeasureSpec.makeMeasureSpec(i3, 1073741824), i2);
        }
        if (c2 || this.f81d == null) {
            return;
        }
        super.setPadding((getMeasuredWidth() - this.f81d.getBounds().width()) / 2, getPaddingTop(), getPaddingRight(), getPaddingBottom());
    }

    @Override // android.widget.TextView, android.view.View
    public void onRestoreInstanceState(Parcelable parcelable) {
        super.onRestoreInstanceState(null);
    }

    @Override // android.widget.TextView, android.view.View
    public boolean onTouchEvent(MotionEvent motionEvent) {
        h0 h0Var;
        if (this.f79b.hasSubMenu() && (h0Var = this.f83f) != null && h0Var.onTouch(this, motionEvent)) {
            return true;
        }
        return super.onTouchEvent(motionEvent);
    }

    public void setCheckable(boolean z) {
    }

    public void setChecked(boolean z) {
    }

    public void setExpandedFormat(boolean z) {
        if (this.i != z) {
            this.i = z;
            i iVar = this.f79b;
            if (iVar != null) {
                iVar.n.onItemActionRequestChanged(iVar);
            }
        }
    }

    public void setIcon(Drawable drawable) {
        this.f81d = drawable;
        if (drawable != null) {
            int intrinsicWidth = drawable.getIntrinsicWidth();
            int intrinsicHeight = drawable.getIntrinsicHeight();
            int i = this.l;
            if (intrinsicWidth > i) {
                intrinsicHeight = (int) (intrinsicHeight * (i / intrinsicWidth));
                intrinsicWidth = i;
            }
            if (intrinsicHeight > i) {
                intrinsicWidth = (int) (intrinsicWidth * (i / intrinsicHeight));
            } else {
                i = intrinsicHeight;
            }
            drawable.setBounds(0, 0, intrinsicWidth, i);
        }
        setCompoundDrawables(drawable, null, null, null);
        e();
    }

    public void setItemInvoker(g.b bVar) {
        this.f82e = bVar;
    }

    @Override // android.widget.TextView, android.view.View
    public void setPadding(int i, int i2, int i3, int i4) {
        this.k = i;
        super.setPadding(i, i2, i3, i4);
    }

    public void setPopupCallback(b bVar) {
        this.f84g = bVar;
    }

    public void setTitle(CharSequence charSequence) {
        this.f80c = charSequence;
        e();
    }
}