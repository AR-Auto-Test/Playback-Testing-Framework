package b.b.h;

import android.content.Context;
import android.content.DialogInterface;
import android.content.res.ColorStateList;
import android.content.res.Resources;
import android.content.res.TypedArray;
import android.database.DataSetObserver;
import android.graphics.PorterDuff;
import android.graphics.Rect;
import android.graphics.drawable.Drawable;
import android.os.Parcel;
import android.os.Parcelable;
import android.util.AttributeSet;
import android.util.Log;
import android.view.MotionEvent;
import android.view.View;
import android.view.ViewGroup;
import android.view.ViewTreeObserver;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.ListAdapter;
import android.widget.ListView;
import android.widget.PopupWindow;
import android.widget.Spinner;
import android.widget.SpinnerAdapter;
import android.widget.ThemedSpinnerAdapter;
import b.b.c.g;
import com.ibosoninnov.unitear.R;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicInteger;

/* compiled from: AppCompatSpinner.java */
/* loaded from: classes.dex */
public class w extends Spinner {

    /* renamed from: b  reason: collision with root package name */
    public static final int[] f934b = {16843505};

    /* renamed from: c  reason: collision with root package name */
    public final b.b.h.e f935c;

    /* renamed from: d  reason: collision with root package name */
    public final Context f936d;

    /* renamed from: e  reason: collision with root package name */
    public h0 f937e;

    /* renamed from: f  reason: collision with root package name */
    public SpinnerAdapter f938f;

    /* renamed from: g  reason: collision with root package name */
    public final boolean f939g;

    /* renamed from: h  reason: collision with root package name */
    public f f940h;
    public int i;
    public final Rect j;

    /* compiled from: AppCompatSpinner.java */
    /* loaded from: classes.dex */
    public class a implements ViewTreeObserver.OnGlobalLayoutListener {
        public a() {
        }

        @Override // android.view.ViewTreeObserver.OnGlobalLayoutListener
        public void onGlobalLayout() {
            if (!w.this.getInternalPopup().a()) {
                w.this.b();
            }
            ViewTreeObserver viewTreeObserver = w.this.getViewTreeObserver();
            if (viewTreeObserver != null) {
                viewTreeObserver.removeOnGlobalLayoutListener(this);
            }
        }
    }

    /* compiled from: AppCompatSpinner.java */
    /* loaded from: classes.dex */
    public class b implements f, DialogInterface.OnClickListener {

        /* renamed from: b  reason: collision with root package name */
        public b.b.c.g f942b;

        /* renamed from: c  reason: collision with root package name */
        public ListAdapter f943c;

        /* renamed from: d  reason: collision with root package name */
        public CharSequence f944d;

        public b() {
        }

        @Override // b.b.h.w.f
        public boolean a() {
            b.b.c.g gVar = this.f942b;
            if (gVar != null) {
                return gVar.isShowing();
            }
            return false;
        }

        @Override // b.b.h.w.f
        public int b() {
            return 0;
        }

        @Override // b.b.h.w.f
        public void d(int i) {
            Log.e("AppCompatSpinner", "Cannot set horizontal offset for MODE_DIALOG, ignoring");
        }

        @Override // b.b.h.w.f
        public void dismiss() {
            b.b.c.g gVar = this.f942b;
            if (gVar != null) {
                gVar.dismiss();
                this.f942b = null;
            }
        }

        @Override // b.b.h.w.f
        public CharSequence e() {
            return this.f944d;
        }

        @Override // b.b.h.w.f
        public Drawable g() {
            return null;
        }

        @Override // b.b.h.w.f
        public void i(CharSequence charSequence) {
            this.f944d = charSequence;
        }

        @Override // b.b.h.w.f
        public void j(int i) {
            Log.e("AppCompatSpinner", "Cannot set vertical offset for MODE_DIALOG, ignoring");
        }

        @Override // b.b.h.w.f
        public void k(int i) {
            Log.e("AppCompatSpinner", "Cannot set horizontal (original) offset for MODE_DIALOG, ignoring");
        }

        @Override // b.b.h.w.f
        public void l(int i, int i2) {
            if (this.f943c == null) {
                return;
            }
            g.a aVar = new g.a(w.this.getPopupContext());
            CharSequence charSequence = this.f944d;
            if (charSequence != null) {
                aVar.setTitle(charSequence);
            }
            b.b.c.g create = aVar.setSingleChoiceItems(this.f943c, w.this.getSelectedItemPosition(), this).create();
            this.f942b = create;
            ListView listView = create.f564b.f65g;
            listView.setTextDirection(i);
            listView.setTextAlignment(i2);
            this.f942b.show();
        }

        @Override // b.b.h.w.f
        public int m() {
            return 0;
        }

        @Override // b.b.h.w.f
        public void n(ListAdapter listAdapter) {
            this.f943c = listAdapter;
        }

        @Override // android.content.DialogInterface.OnClickListener
        public void onClick(DialogInterface dialogInterface, int i) {
            w.this.setSelection(i);
            if (w.this.getOnItemClickListener() != null) {
                w.this.performItemClick(null, i, this.f943c.getItemId(i));
            }
            b.b.c.g gVar = this.f942b;
            if (gVar != null) {
                gVar.dismiss();
                this.f942b = null;
            }
        }

        @Override // b.b.h.w.f
        public void setBackgroundDrawable(Drawable drawable) {
            Log.e("AppCompatSpinner", "Cannot set popup background for MODE_DIALOG, ignoring");
        }
    }

    /* compiled from: AppCompatSpinner.java */
    /* loaded from: classes.dex */
    public static class c implements ListAdapter, SpinnerAdapter {

        /* renamed from: b  reason: collision with root package name */
        public SpinnerAdapter f946b;

        /* renamed from: c  reason: collision with root package name */
        public ListAdapter f947c;

        public c(SpinnerAdapter spinnerAdapter, Resources.Theme theme) {
            this.f946b = spinnerAdapter;
            if (spinnerAdapter instanceof ListAdapter) {
                this.f947c = (ListAdapter) spinnerAdapter;
            }
            if (theme != null) {
                if (spinnerAdapter instanceof ThemedSpinnerAdapter) {
                    ThemedSpinnerAdapter themedSpinnerAdapter = (ThemedSpinnerAdapter) spinnerAdapter;
                    if (themedSpinnerAdapter.getDropDownViewTheme() != theme) {
                        themedSpinnerAdapter.setDropDownViewTheme(theme);
                    }
                } else if (spinnerAdapter instanceof u0) {
                    u0 u0Var = (u0) spinnerAdapter;
                    if (u0Var.getDropDownViewTheme() == null) {
                        u0Var.setDropDownViewTheme(theme);
                    }
                }
            }
        }

        @Override // android.widget.ListAdapter
        public boolean areAllItemsEnabled() {
            ListAdapter listAdapter = this.f947c;
            if (listAdapter != null) {
                return listAdapter.areAllItemsEnabled();
            }
            return true;
        }

        @Override // android.widget.Adapter
        public int getCount() {
            SpinnerAdapter spinnerAdapter = this.f946b;
            if (spinnerAdapter == null) {
                return 0;
            }
            return spinnerAdapter.getCount();
        }

        @Override // android.widget.SpinnerAdapter
        public View getDropDownView(int i, View view, ViewGroup viewGroup) {
            SpinnerAdapter spinnerAdapter = this.f946b;
            if (spinnerAdapter == null) {
                return null;
            }
            return spinnerAdapter.getDropDownView(i, view, viewGroup);
        }

        @Override // android.widget.Adapter
        public Object getItem(int i) {
            SpinnerAdapter spinnerAdapter = this.f946b;
            if (spinnerAdapter == null) {
                return null;
            }
            return spinnerAdapter.getItem(i);
        }

        @Override // android.widget.Adapter
        public long getItemId(int i) {
            SpinnerAdapter spinnerAdapter = this.f946b;
            if (spinnerAdapter == null) {
                return -1L;
            }
            return spinnerAdapter.getItemId(i);
        }

        @Override // android.widget.Adapter
        public int getItemViewType(int i) {
            return 0;
        }

        @Override // android.widget.Adapter
        public View getView(int i, View view, ViewGroup viewGroup) {
            SpinnerAdapter spinnerAdapter = this.f946b;
            if (spinnerAdapter == null) {
                return null;
            }
            return spinnerAdapter.getDropDownView(i, view, viewGroup);
        }

        @Override // android.widget.Adapter
        public int getViewTypeCount() {
            return 1;
        }

        @Override // android.widget.Adapter
        public boolean hasStableIds() {
            SpinnerAdapter spinnerAdapter = this.f946b;
            return spinnerAdapter != null && spinnerAdapter.hasStableIds();
        }

        @Override // android.widget.Adapter
        public boolean isEmpty() {
            return getCount() == 0;
        }

        @Override // android.widget.ListAdapter
        public boolean isEnabled(int i) {
            ListAdapter listAdapter = this.f947c;
            if (listAdapter != null) {
                return listAdapter.isEnabled(i);
            }
            return true;
        }

        @Override // android.widget.Adapter
        public void registerDataSetObserver(DataSetObserver dataSetObserver) {
            SpinnerAdapter spinnerAdapter = this.f946b;
            if (spinnerAdapter != null) {
                spinnerAdapter.registerDataSetObserver(dataSetObserver);
            }
        }

        @Override // android.widget.Adapter
        public void unregisterDataSetObserver(DataSetObserver dataSetObserver) {
            SpinnerAdapter spinnerAdapter = this.f946b;
            if (spinnerAdapter != null) {
                spinnerAdapter.unregisterDataSetObserver(dataSetObserver);
            }
        }
    }

    /* compiled from: AppCompatSpinner.java */
    /* loaded from: classes.dex */
    public class d extends k0 implements f {
        public CharSequence D;
        public ListAdapter E;
        public final Rect F;
        public int G;

        /* compiled from: AppCompatSpinner.java */
        /* loaded from: classes.dex */
        public class a implements AdapterView.OnItemClickListener {
            public a(w wVar) {
            }

            @Override // android.widget.AdapterView.OnItemClickListener
            public void onItemClick(AdapterView<?> adapterView, View view, int i, long j) {
                w.this.setSelection(i);
                if (w.this.getOnItemClickListener() != null) {
                    d dVar = d.this;
                    w.this.performItemClick(view, i, dVar.E.getItemId(i));
                }
                d.this.dismiss();
            }
        }

        /* compiled from: AppCompatSpinner.java */
        /* loaded from: classes.dex */
        public class b implements ViewTreeObserver.OnGlobalLayoutListener {
            public b() {
            }

            @Override // android.view.ViewTreeObserver.OnGlobalLayoutListener
            public void onGlobalLayout() {
                d dVar = d.this;
                w wVar = w.this;
                Objects.requireNonNull(dVar);
                AtomicInteger atomicInteger = b.j.j.q.f2214a;
                if (!(wVar.isAttachedToWindow() && wVar.getGlobalVisibleRect(dVar.F))) {
                    d.this.dismiss();
                    return;
                }
                d.this.r();
                d.this.show();
            }
        }

        /* compiled from: AppCompatSpinner.java */
        /* loaded from: classes.dex */
        public class c implements PopupWindow.OnDismissListener {

            /* renamed from: b  reason: collision with root package name */
            public final /* synthetic */ ViewTreeObserver.OnGlobalLayoutListener f950b;

            public c(ViewTreeObserver.OnGlobalLayoutListener onGlobalLayoutListener) {
                this.f950b = onGlobalLayoutListener;
            }

            @Override // android.widget.PopupWindow.OnDismissListener
            public void onDismiss() {
                ViewTreeObserver viewTreeObserver = w.this.getViewTreeObserver();
                if (viewTreeObserver != null) {
                    viewTreeObserver.removeGlobalOnLayoutListener(this.f950b);
                }
            }
        }

        public d(Context context, AttributeSet attributeSet, int i) {
            super(context, attributeSet, i, 0);
            this.F = new Rect();
            this.s = w.this;
            q(true);
            this.q = 0;
            this.t = new a(w.this);
        }

        @Override // b.b.h.w.f
        public CharSequence e() {
            return this.D;
        }

        @Override // b.b.h.w.f
        public void i(CharSequence charSequence) {
            this.D = charSequence;
        }

        @Override // b.b.h.w.f
        public void k(int i) {
            this.G = i;
        }

        @Override // b.b.h.w.f
        public void l(int i, int i2) {
            ViewTreeObserver viewTreeObserver;
            boolean a2 = a();
            r();
            this.C.setInputMethodMode(2);
            show();
            f0 f0Var = this.f876f;
            f0Var.setChoiceMode(1);
            f0Var.setTextDirection(i);
            f0Var.setTextAlignment(i2);
            int selectedItemPosition = w.this.getSelectedItemPosition();
            f0 f0Var2 = this.f876f;
            if (a() && f0Var2 != null) {
                f0Var2.setListSelectionHidden(false);
                f0Var2.setSelection(selectedItemPosition);
                if (f0Var2.getChoiceMode() != 0) {
                    f0Var2.setItemChecked(selectedItemPosition, true);
                }
            }
            if (a2 || (viewTreeObserver = w.this.getViewTreeObserver()) == null) {
                return;
            }
            b bVar = new b();
            viewTreeObserver.addOnGlobalLayoutListener(bVar);
            this.C.setOnDismissListener(new c(bVar));
        }

        @Override // b.b.h.k0, b.b.h.w.f
        public void n(ListAdapter listAdapter) {
            super.n(listAdapter);
            this.E = listAdapter;
        }

        public void r() {
            int i;
            Drawable g2 = g();
            int i2 = 0;
            if (g2 != null) {
                g2.getPadding(w.this.j);
                i2 = e1.b(w.this) ? w.this.j.right : -w.this.j.left;
            } else {
                Rect rect = w.this.j;
                rect.right = 0;
                rect.left = 0;
            }
            int paddingLeft = w.this.getPaddingLeft();
            int paddingRight = w.this.getPaddingRight();
            int width = w.this.getWidth();
            w wVar = w.this;
            int i3 = wVar.i;
            if (i3 == -2) {
                int a2 = wVar.a((SpinnerAdapter) this.E, g());
                int i4 = w.this.getContext().getResources().getDisplayMetrics().widthPixels;
                Rect rect2 = w.this.j;
                int i5 = (i4 - rect2.left) - rect2.right;
                if (a2 > i5) {
                    a2 = i5;
                }
                p(Math.max(a2, (width - paddingLeft) - paddingRight));
            } else if (i3 == -1) {
                p((width - paddingLeft) - paddingRight);
            } else {
                p(i3);
            }
            if (e1.b(w.this)) {
                i = (((width - paddingRight) - this.f878h) - this.G) + i2;
            } else {
                i = paddingLeft + this.G + i2;
            }
            this.i = i;
        }
    }

    /* compiled from: AppCompatSpinner.java */
    /* loaded from: classes.dex */
    public static class e extends View.BaseSavedState {
        public static final Parcelable.Creator<e> CREATOR = new a();

        /* renamed from: b  reason: collision with root package name */
        public boolean f952b;

        /* compiled from: AppCompatSpinner.java */
        /* loaded from: classes.dex */
        public class a implements Parcelable.Creator<e> {
            /* JADX DEBUG: Return type fixed from 'java.lang.Object' to match base method */
            @Override // android.os.Parcelable.Creator
            public e createFromParcel(Parcel parcel) {
                return new e(parcel);
            }

            /* JADX DEBUG: Return type fixed from 'java.lang.Object[]' to match base method */
            @Override // android.os.Parcelable.Creator
            public e[] newArray(int i) {
                return new e[i];
            }
        }

        public e(Parcelable parcelable) {
            super(parcelable);
        }

        @Override // android.view.View.BaseSavedState, android.view.AbsSavedState, android.os.Parcelable
        public void writeToParcel(Parcel parcel, int i) {
            super.writeToParcel(parcel, i);
            parcel.writeByte(this.f952b ? (byte) 1 : (byte) 0);
        }

        public e(Parcel parcel) {
            super(parcel);
            this.f952b = parcel.readByte() != 0;
        }
    }

    /* compiled from: AppCompatSpinner.java */
    /* loaded from: classes.dex */
    public interface f {
        boolean a();

        int b();

        void d(int i);

        void dismiss();

        CharSequence e();

        Drawable g();

        void i(CharSequence charSequence);

        void j(int i);

        void k(int i);

        void l(int i, int i2);

        int m();

        void n(ListAdapter listAdapter);

        void setBackgroundDrawable(Drawable drawable);
    }

    /* JADX WARN: Code restructure failed: missing block: B:21:0x0056, code lost:
        if (r4 == null) goto L13;
     */
    /* JADX WARN: Removed duplicated region for block: B:38:0x00d2  */
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public w(Context context, AttributeSet attributeSet, int i) {
        super(context, attributeSet, i);
        Exception e2;
        TypedArray typedArray;
        this.j = new Rect();
        t0.a(this, getContext());
        TypedArray obtainStyledAttributes = context.obtainStyledAttributes(attributeSet, b.b.b.v, i, 0);
        this.f935c = new b.b.h.e(this);
        int resourceId = obtainStyledAttributes.getResourceId(4, 0);
        if (resourceId != 0) {
            this.f936d = new b.b.g.c(context, resourceId);
        } else {
            this.f936d = context;
        }
        TypedArray typedArray2 = null;
        int i2 = -1;
        try {
            try {
                typedArray = context.obtainStyledAttributes(attributeSet, f934b, i, 0);
            } catch (Exception e3) {
                e2 = e3;
                typedArray = null;
            } catch (Throwable th) {
                th = th;
                if (typedArray2 != null) {
                }
                throw th;
            }
            try {
                if (typedArray.hasValue(0)) {
                    i2 = typedArray.getInt(0, 0);
                }
            } catch (Exception e4) {
                e2 = e4;
                Log.i("AppCompatSpinner", "Could not read android:spinnerMode", e2);
            }
            typedArray.recycle();
            if (i2 == 0) {
                b bVar = new b();
                this.f940h = bVar;
                bVar.i(obtainStyledAttributes.getString(2));
            } else if (i2 == 1) {
                d dVar = new d(this.f936d, attributeSet, i);
                y0 r = y0.r(this.f936d, attributeSet, b.b.b.v, i, 0);
                this.i = r.l(3, -2);
                dVar.C.setBackgroundDrawable(r.g(1));
                dVar.D = obtainStyledAttributes.getString(2);
                r.f972b.recycle();
                this.f940h = dVar;
                this.f937e = new v(this, this, dVar);
            }
            CharSequence[] textArray = obtainStyledAttributes.getTextArray(0);
            if (textArray != null) {
                ArrayAdapter arrayAdapter = new ArrayAdapter(context, 17367048, textArray);
                arrayAdapter.setDropDownViewResource(R.layout.support_simple_spinner_dropdown_item);
                setAdapter((SpinnerAdapter) arrayAdapter);
            }
            obtainStyledAttributes.recycle();
            this.f939g = true;
            SpinnerAdapter spinnerAdapter = this.f938f;
            if (spinnerAdapter != null) {
                setAdapter(spinnerAdapter);
                this.f938f = null;
            }
            this.f935c.d(attributeSet, i);
        } catch (Throwable th2) {
            th = th2;
            typedArray2 = typedArray;
            if (typedArray2 != null) {
                typedArray2.recycle();
            }
            throw th;
        }
    }

    public int a(SpinnerAdapter spinnerAdapter, Drawable drawable) {
        int i = 0;
        if (spinnerAdapter == null) {
            return 0;
        }
        int makeMeasureSpec = View.MeasureSpec.makeMeasureSpec(getMeasuredWidth(), 0);
        int makeMeasureSpec2 = View.MeasureSpec.makeMeasureSpec(getMeasuredHeight(), 0);
        int max = Math.max(0, getSelectedItemPosition());
        int min = Math.min(spinnerAdapter.getCount(), max + 15);
        View view = null;
        int i2 = 0;
        for (int max2 = Math.max(0, max - (15 - (min - max))); max2 < min; max2++) {
            int itemViewType = spinnerAdapter.getItemViewType(max2);
            if (itemViewType != i) {
                view = null;
                i = itemViewType;
            }
            view = spinnerAdapter.getView(max2, view, this);
            if (view.getLayoutParams() == null) {
                view.setLayoutParams(new ViewGroup.LayoutParams(-2, -2));
            }
            view.measure(makeMeasureSpec, makeMeasureSpec2);
            i2 = Math.max(i2, view.getMeasuredWidth());
        }
        if (drawable != null) {
            drawable.getPadding(this.j);
            Rect rect = this.j;
            return i2 + rect.left + rect.right;
        }
        return i2;
    }

    public void b() {
        this.f940h.l(getTextDirection(), getTextAlignment());
    }

    @Override // android.view.ViewGroup, android.view.View
    public void drawableStateChanged() {
        super.drawableStateChanged();
        b.b.h.e eVar = this.f935c;
        if (eVar != null) {
            eVar.a();
        }
    }

    @Override // android.widget.Spinner
    public int getDropDownHorizontalOffset() {
        f fVar = this.f940h;
        if (fVar != null) {
            return fVar.b();
        }
        return super.getDropDownHorizontalOffset();
    }

    @Override // android.widget.Spinner
    public int getDropDownVerticalOffset() {
        f fVar = this.f940h;
        if (fVar != null) {
            return fVar.m();
        }
        return super.getDropDownVerticalOffset();
    }

    @Override // android.widget.Spinner
    public int getDropDownWidth() {
        if (this.f940h != null) {
            return this.i;
        }
        return super.getDropDownWidth();
    }

    public final f getInternalPopup() {
        return this.f940h;
    }

    @Override // android.widget.Spinner
    public Drawable getPopupBackground() {
        f fVar = this.f940h;
        if (fVar != null) {
            return fVar.g();
        }
        return super.getPopupBackground();
    }

    @Override // android.widget.Spinner
    public Context getPopupContext() {
        return this.f936d;
    }

    @Override // android.widget.Spinner
    public CharSequence getPrompt() {
        f fVar = this.f940h;
        return fVar != null ? fVar.e() : super.getPrompt();
    }

    public ColorStateList getSupportBackgroundTintList() {
        b.b.h.e eVar = this.f935c;
        if (eVar != null) {
            return eVar.b();
        }
        return null;
    }

    public PorterDuff.Mode getSupportBackgroundTintMode() {
        b.b.h.e eVar = this.f935c;
        if (eVar != null) {
            return eVar.c();
        }
        return null;
    }

    @Override // android.widget.Spinner, android.widget.AdapterView, android.view.ViewGroup, android.view.View
    public void onDetachedFromWindow() {
        super.onDetachedFromWindow();
        f fVar = this.f940h;
        if (fVar == null || !fVar.a()) {
            return;
        }
        this.f940h.dismiss();
    }

    @Override // android.widget.Spinner, android.widget.AbsSpinner, android.view.View
    public void onMeasure(int i, int i2) {
        super.onMeasure(i, i2);
        if (this.f940h == null || View.MeasureSpec.getMode(i) != Integer.MIN_VALUE) {
            return;
        }
        setMeasuredDimension(Math.min(Math.max(getMeasuredWidth(), a(getAdapter(), getBackground())), View.MeasureSpec.getSize(i)), getMeasuredHeight());
    }

    @Override // android.widget.Spinner, android.widget.AbsSpinner, android.view.View
    public void onRestoreInstanceState(Parcelable parcelable) {
        ViewTreeObserver viewTreeObserver;
        e eVar = (e) parcelable;
        super.onRestoreInstanceState(eVar.getSuperState());
        if (!eVar.f952b || (viewTreeObserver = getViewTreeObserver()) == null) {
            return;
        }
        viewTreeObserver.addOnGlobalLayoutListener(new a());
    }

    @Override // android.widget.Spinner, android.widget.AbsSpinner, android.view.View
    public Parcelable onSaveInstanceState() {
        e eVar = new e(super.onSaveInstanceState());
        f fVar = this.f940h;
        eVar.f952b = fVar != null && fVar.a();
        return eVar;
    }

    @Override // android.widget.Spinner, android.view.View
    public boolean onTouchEvent(MotionEvent motionEvent) {
        h0 h0Var = this.f937e;
        if (h0Var == null || !h0Var.onTouch(this, motionEvent)) {
            return super.onTouchEvent(motionEvent);
        }
        return true;
    }

    @Override // android.widget.Spinner, android.view.View
    public boolean performClick() {
        f fVar = this.f940h;
        if (fVar != null) {
            if (fVar.a()) {
                return true;
            }
            b();
            return true;
        }
        return super.performClick();
    }

    @Override // android.view.View
    public void setBackgroundDrawable(Drawable drawable) {
        super.setBackgroundDrawable(drawable);
        b.b.h.e eVar = this.f935c;
        if (eVar != null) {
            eVar.e();
        }
    }

    @Override // android.view.View
    public void setBackgroundResource(int i) {
        super.setBackgroundResource(i);
        b.b.h.e eVar = this.f935c;
        if (eVar != null) {
            eVar.f(i);
        }
    }

    @Override // android.widget.Spinner
    public void setDropDownHorizontalOffset(int i) {
        f fVar = this.f940h;
        if (fVar != null) {
            fVar.k(i);
            this.f940h.d(i);
            return;
        }
        super.setDropDownHorizontalOffset(i);
    }

    @Override // android.widget.Spinner
    public void setDropDownVerticalOffset(int i) {
        f fVar = this.f940h;
        if (fVar != null) {
            fVar.j(i);
        } else {
            super.setDropDownVerticalOffset(i);
        }
    }

    @Override // android.widget.Spinner
    public void setDropDownWidth(int i) {
        if (this.f940h != null) {
            this.i = i;
        } else {
            super.setDropDownWidth(i);
        }
    }

    @Override // android.widget.Spinner
    public void setPopupBackgroundDrawable(Drawable drawable) {
        f fVar = this.f940h;
        if (fVar != null) {
            fVar.setBackgroundDrawable(drawable);
        } else {
            super.setPopupBackgroundDrawable(drawable);
        }
    }

    @Override // android.widget.Spinner
    public void setPopupBackgroundResource(int i) {
        setPopupBackgroundDrawable(b.b.d.a.a.a(getPopupContext(), i));
    }

    @Override // android.widget.Spinner
    public void setPrompt(CharSequence charSequence) {
        f fVar = this.f940h;
        if (fVar != null) {
            fVar.i(charSequence);
        } else {
            super.setPrompt(charSequence);
        }
    }

    public void setSupportBackgroundTintList(ColorStateList colorStateList) {
        b.b.h.e eVar = this.f935c;
        if (eVar != null) {
            eVar.h(colorStateList);
        }
    }

    public void setSupportBackgroundTintMode(PorterDuff.Mode mode) {
        b.b.h.e eVar = this.f935c;
        if (eVar != null) {
            eVar.i(mode);
        }
    }

    /* JADX DEBUG: Method merged with bridge method */
    @Override // android.widget.AdapterView
    public void setAdapter(SpinnerAdapter spinnerAdapter) {
        if (!this.f939g) {
            this.f938f = spinnerAdapter;
            return;
        }
        super.setAdapter(spinnerAdapter);
        if (this.f940h != null) {
            Context context = this.f936d;
            if (context == null) {
                context = getContext();
            }
            this.f940h.n(new c(spinnerAdapter, context.getTheme()));
        }
    }
}