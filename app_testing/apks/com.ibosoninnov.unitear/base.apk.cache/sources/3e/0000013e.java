package b.b.c;

import android.content.Context;
import android.content.DialogInterface;
import android.database.Cursor;
import android.graphics.drawable.Drawable;
import android.os.Bundle;
import android.text.TextUtils;
import android.util.TypedValue;
import android.view.ContextThemeWrapper;
import android.view.KeyEvent;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.AdapterView;
import android.widget.Button;
import android.widget.FrameLayout;
import android.widget.ImageView;
import android.widget.ListAdapter;
import android.widget.ListView;
import android.widget.SimpleCursorAdapter;
import android.widget.TextView;
import androidx.appcompat.app.AlertController;
import androidx.core.widget.NestedScrollView;
import b.b.h.i0;
import com.google.android.material.internal.StaticLayoutBuilderCompat;
import com.ibosoninnov.unitear.R;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicInteger;

/* compiled from: AlertDialog.java */
/* loaded from: classes.dex */
public class g extends p implements DialogInterface {

    /* renamed from: b  reason: collision with root package name */
    public final AlertController f564b;

    /* compiled from: AlertDialog.java */
    /* loaded from: classes.dex */
    public static class a {
        private final AlertController.b P;
        private final int mTheme;

        public a(Context context) {
            this(context, g.a(context, 0));
        }

        public g create() {
            int i;
            ListAdapter listAdapter;
            g gVar = new g(this.P.f70a, this.mTheme);
            AlertController.b bVar = this.P;
            AlertController alertController = gVar.f564b;
            View view = bVar.f75f;
            if (view != null) {
                alertController.G = view;
            } else {
                CharSequence charSequence = bVar.f74e;
                if (charSequence != null) {
                    alertController.f63e = charSequence;
                    TextView textView = alertController.E;
                    if (textView != null) {
                        textView.setText(charSequence);
                    }
                }
                Drawable drawable = bVar.f73d;
                if (drawable != null) {
                    alertController.C = drawable;
                    alertController.B = 0;
                    ImageView imageView = alertController.D;
                    if (imageView != null) {
                        imageView.setVisibility(0);
                        alertController.D.setImageDrawable(drawable);
                    }
                }
                int i2 = bVar.f72c;
                if (i2 != 0) {
                    alertController.f(i2);
                }
            }
            CharSequence charSequence2 = bVar.f76g;
            if (charSequence2 != null) {
                alertController.f64f = charSequence2;
                TextView textView2 = alertController.F;
                if (textView2 != null) {
                    textView2.setText(charSequence2);
                }
            }
            CharSequence charSequence3 = bVar.f77h;
            if (charSequence3 != null || bVar.i != null) {
                alertController.e(-1, charSequence3, bVar.j, null, bVar.i);
            }
            CharSequence charSequence4 = bVar.k;
            if (charSequence4 != null || bVar.l != null) {
                alertController.e(-2, charSequence4, bVar.m, null, bVar.l);
            }
            CharSequence charSequence5 = bVar.n;
            if (charSequence5 != null || bVar.o != null) {
                alertController.e(-3, charSequence5, bVar.p, null, bVar.o);
            }
            if (bVar.u != null || bVar.J != null || bVar.v != null) {
                AlertController.RecycleListView recycleListView = (AlertController.RecycleListView) bVar.f71b.inflate(alertController.L, (ViewGroup) null);
                if (bVar.F) {
                    if (bVar.J == null) {
                        listAdapter = new c(bVar, bVar.f70a, alertController.M, 16908308, bVar.u, recycleListView);
                    } else {
                        listAdapter = new d(bVar, bVar.f70a, bVar.J, false, recycleListView, alertController);
                    }
                } else {
                    if (bVar.G) {
                        i = alertController.N;
                    } else {
                        i = alertController.O;
                    }
                    int i3 = i;
                    if (bVar.J != null) {
                        listAdapter = new SimpleCursorAdapter(bVar.f70a, i3, bVar.J, new String[]{bVar.K}, new int[]{16908308});
                    } else {
                        listAdapter = bVar.v;
                        if (listAdapter == null) {
                            listAdapter = new AlertController.d(bVar.f70a, i3, 16908308, bVar.u);
                        }
                    }
                }
                alertController.H = listAdapter;
                alertController.I = bVar.H;
                if (bVar.w != null) {
                    recycleListView.setOnItemClickListener(new e(bVar, alertController));
                } else if (bVar.I != null) {
                    recycleListView.setOnItemClickListener(new f(bVar, recycleListView, alertController));
                }
                AdapterView.OnItemSelectedListener onItemSelectedListener = bVar.M;
                if (onItemSelectedListener != null) {
                    recycleListView.setOnItemSelectedListener(onItemSelectedListener);
                }
                if (bVar.G) {
                    recycleListView.setChoiceMode(1);
                } else if (bVar.F) {
                    recycleListView.setChoiceMode(2);
                }
                alertController.f65g = recycleListView;
            }
            View view2 = bVar.y;
            if (view2 != null) {
                if (bVar.D) {
                    int i4 = bVar.z;
                    int i5 = bVar.A;
                    int i6 = bVar.B;
                    int i7 = bVar.C;
                    alertController.f66h = view2;
                    alertController.i = 0;
                    alertController.n = true;
                    alertController.j = i4;
                    alertController.k = i5;
                    alertController.l = i6;
                    alertController.m = i7;
                } else {
                    alertController.f66h = view2;
                    alertController.i = 0;
                    alertController.n = false;
                }
            } else {
                int i8 = bVar.x;
                if (i8 != 0) {
                    alertController.f66h = null;
                    alertController.i = i8;
                    alertController.n = false;
                }
            }
            gVar.setCancelable(this.P.q);
            if (this.P.q) {
                gVar.setCanceledOnTouchOutside(true);
            }
            gVar.setOnCancelListener(this.P.r);
            gVar.setOnDismissListener(this.P.s);
            DialogInterface.OnKeyListener onKeyListener = this.P.t;
            if (onKeyListener != null) {
                gVar.setOnKeyListener(onKeyListener);
            }
            return gVar;
        }

        public Context getContext() {
            return this.P.f70a;
        }

        public a setAdapter(ListAdapter listAdapter, DialogInterface.OnClickListener onClickListener) {
            AlertController.b bVar = this.P;
            bVar.v = listAdapter;
            bVar.w = onClickListener;
            return this;
        }

        public a setCancelable(boolean z) {
            this.P.q = z;
            return this;
        }

        public a setCursor(Cursor cursor, DialogInterface.OnClickListener onClickListener, String str) {
            AlertController.b bVar = this.P;
            bVar.J = cursor;
            bVar.K = str;
            bVar.w = onClickListener;
            return this;
        }

        public a setCustomTitle(View view) {
            this.P.f75f = view;
            return this;
        }

        public a setIcon(int i) {
            this.P.f72c = i;
            return this;
        }

        public a setIconAttribute(int i) {
            TypedValue typedValue = new TypedValue();
            this.P.f70a.getTheme().resolveAttribute(i, typedValue, true);
            this.P.f72c = typedValue.resourceId;
            return this;
        }

        @Deprecated
        public a setInverseBackgroundForced(boolean z) {
            Objects.requireNonNull(this.P);
            return this;
        }

        public a setItems(int i, DialogInterface.OnClickListener onClickListener) {
            AlertController.b bVar = this.P;
            bVar.u = bVar.f70a.getResources().getTextArray(i);
            this.P.w = onClickListener;
            return this;
        }

        public a setMessage(int i) {
            AlertController.b bVar = this.P;
            bVar.f76g = bVar.f70a.getText(i);
            return this;
        }

        public a setMultiChoiceItems(int i, boolean[] zArr, DialogInterface.OnMultiChoiceClickListener onMultiChoiceClickListener) {
            AlertController.b bVar = this.P;
            bVar.u = bVar.f70a.getResources().getTextArray(i);
            AlertController.b bVar2 = this.P;
            bVar2.I = onMultiChoiceClickListener;
            bVar2.E = zArr;
            bVar2.F = true;
            return this;
        }

        public a setNegativeButton(int i, DialogInterface.OnClickListener onClickListener) {
            AlertController.b bVar = this.P;
            bVar.k = bVar.f70a.getText(i);
            this.P.m = onClickListener;
            return this;
        }

        public a setNegativeButtonIcon(Drawable drawable) {
            this.P.l = drawable;
            return this;
        }

        public a setNeutralButton(int i, DialogInterface.OnClickListener onClickListener) {
            AlertController.b bVar = this.P;
            bVar.n = bVar.f70a.getText(i);
            this.P.p = onClickListener;
            return this;
        }

        public a setNeutralButtonIcon(Drawable drawable) {
            this.P.o = drawable;
            return this;
        }

        public a setOnCancelListener(DialogInterface.OnCancelListener onCancelListener) {
            this.P.r = onCancelListener;
            return this;
        }

        public a setOnDismissListener(DialogInterface.OnDismissListener onDismissListener) {
            this.P.s = onDismissListener;
            return this;
        }

        public a setOnItemSelectedListener(AdapterView.OnItemSelectedListener onItemSelectedListener) {
            this.P.M = onItemSelectedListener;
            return this;
        }

        public a setOnKeyListener(DialogInterface.OnKeyListener onKeyListener) {
            this.P.t = onKeyListener;
            return this;
        }

        public a setPositiveButton(int i, DialogInterface.OnClickListener onClickListener) {
            AlertController.b bVar = this.P;
            bVar.f77h = bVar.f70a.getText(i);
            this.P.j = onClickListener;
            return this;
        }

        public a setPositiveButtonIcon(Drawable drawable) {
            this.P.i = drawable;
            return this;
        }

        public a setRecycleOnMeasureEnabled(boolean z) {
            Objects.requireNonNull(this.P);
            return this;
        }

        public a setSingleChoiceItems(int i, int i2, DialogInterface.OnClickListener onClickListener) {
            AlertController.b bVar = this.P;
            bVar.u = bVar.f70a.getResources().getTextArray(i);
            AlertController.b bVar2 = this.P;
            bVar2.w = onClickListener;
            bVar2.H = i2;
            bVar2.G = true;
            return this;
        }

        public a setTitle(int i) {
            AlertController.b bVar = this.P;
            bVar.f74e = bVar.f70a.getText(i);
            return this;
        }

        public a setView(int i) {
            AlertController.b bVar = this.P;
            bVar.y = null;
            bVar.x = i;
            bVar.D = false;
            return this;
        }

        public g show() {
            g create = create();
            create.show();
            return create;
        }

        public a(Context context, int i) {
            this.P = new AlertController.b(new ContextThemeWrapper(context, g.a(context, i)));
            this.mTheme = i;
        }

        public a setIcon(Drawable drawable) {
            this.P.f73d = drawable;
            return this;
        }

        public a setMessage(CharSequence charSequence) {
            this.P.f76g = charSequence;
            return this;
        }

        public a setTitle(CharSequence charSequence) {
            this.P.f74e = charSequence;
            return this;
        }

        public a setItems(CharSequence[] charSequenceArr, DialogInterface.OnClickListener onClickListener) {
            AlertController.b bVar = this.P;
            bVar.u = charSequenceArr;
            bVar.w = onClickListener;
            return this;
        }

        public a setNegativeButton(CharSequence charSequence, DialogInterface.OnClickListener onClickListener) {
            AlertController.b bVar = this.P;
            bVar.k = charSequence;
            bVar.m = onClickListener;
            return this;
        }

        public a setNeutralButton(CharSequence charSequence, DialogInterface.OnClickListener onClickListener) {
            AlertController.b bVar = this.P;
            bVar.n = charSequence;
            bVar.p = onClickListener;
            return this;
        }

        public a setPositiveButton(CharSequence charSequence, DialogInterface.OnClickListener onClickListener) {
            AlertController.b bVar = this.P;
            bVar.f77h = charSequence;
            bVar.j = onClickListener;
            return this;
        }

        public a setView(View view) {
            AlertController.b bVar = this.P;
            bVar.y = view;
            bVar.x = 0;
            bVar.D = false;
            return this;
        }

        public a setMultiChoiceItems(CharSequence[] charSequenceArr, boolean[] zArr, DialogInterface.OnMultiChoiceClickListener onMultiChoiceClickListener) {
            AlertController.b bVar = this.P;
            bVar.u = charSequenceArr;
            bVar.I = onMultiChoiceClickListener;
            bVar.E = zArr;
            bVar.F = true;
            return this;
        }

        public a setSingleChoiceItems(Cursor cursor, int i, String str, DialogInterface.OnClickListener onClickListener) {
            AlertController.b bVar = this.P;
            bVar.J = cursor;
            bVar.w = onClickListener;
            bVar.H = i;
            bVar.K = str;
            bVar.G = true;
            return this;
        }

        @Deprecated
        public a setView(View view, int i, int i2, int i3, int i4) {
            AlertController.b bVar = this.P;
            bVar.y = view;
            bVar.x = 0;
            bVar.D = true;
            bVar.z = i;
            bVar.A = i2;
            bVar.B = i3;
            bVar.C = i4;
            return this;
        }

        public a setMultiChoiceItems(Cursor cursor, String str, String str2, DialogInterface.OnMultiChoiceClickListener onMultiChoiceClickListener) {
            AlertController.b bVar = this.P;
            bVar.J = cursor;
            bVar.I = onMultiChoiceClickListener;
            bVar.L = str;
            bVar.K = str2;
            bVar.F = true;
            return this;
        }

        public a setSingleChoiceItems(CharSequence[] charSequenceArr, int i, DialogInterface.OnClickListener onClickListener) {
            AlertController.b bVar = this.P;
            bVar.u = charSequenceArr;
            bVar.w = onClickListener;
            bVar.H = i;
            bVar.G = true;
            return this;
        }

        public a setSingleChoiceItems(ListAdapter listAdapter, int i, DialogInterface.OnClickListener onClickListener) {
            AlertController.b bVar = this.P;
            bVar.v = listAdapter;
            bVar.w = onClickListener;
            bVar.H = i;
            bVar.G = true;
            return this;
        }
    }

    public g(Context context, int i) {
        super(context, a(context, i));
        this.f564b = new AlertController(getContext(), this, getWindow());
    }

    public static int a(Context context, int i) {
        if (((i >>> 24) & 255) >= 1) {
            return i;
        }
        TypedValue typedValue = new TypedValue();
        context.getTheme().resolveAttribute(R.attr.alertDialogTheme, typedValue, true);
        return typedValue.resourceId;
    }

    @Override // b.b.c.p, android.app.Dialog
    public void onCreate(Bundle bundle) {
        int i;
        boolean z;
        View view;
        ListAdapter listAdapter;
        View findViewById;
        super.onCreate(bundle);
        AlertController alertController = this.f564b;
        if (alertController.K == 0) {
            i = alertController.J;
        } else {
            i = alertController.J;
        }
        alertController.f60b.setContentView(i);
        View findViewById2 = alertController.f61c.findViewById(R.id.parentPanel);
        View findViewById3 = findViewById2.findViewById(R.id.topPanel);
        View findViewById4 = findViewById2.findViewById(R.id.contentPanel);
        View findViewById5 = findViewById2.findViewById(R.id.buttonPanel);
        ViewGroup viewGroup = (ViewGroup) findViewById2.findViewById(R.id.customPanel);
        View view2 = alertController.f66h;
        if (view2 == null) {
            view2 = alertController.i != 0 ? LayoutInflater.from(alertController.f59a).inflate(alertController.i, viewGroup, false) : null;
        }
        boolean z2 = view2 != null;
        if (!z2 || !AlertController.a(view2)) {
            alertController.f61c.setFlags(131072, 131072);
        }
        if (z2) {
            FrameLayout frameLayout = (FrameLayout) alertController.f61c.findViewById(R.id.custom);
            frameLayout.addView(view2, new ViewGroup.LayoutParams(-1, -1));
            if (alertController.n) {
                frameLayout.setPadding(alertController.j, alertController.k, alertController.l, alertController.m);
            }
            if (alertController.f65g != null) {
                ((i0.a) viewGroup.getLayoutParams()).f860a = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
            }
        } else {
            viewGroup.setVisibility(8);
        }
        View findViewById6 = viewGroup.findViewById(R.id.topPanel);
        View findViewById7 = viewGroup.findViewById(R.id.contentPanel);
        View findViewById8 = viewGroup.findViewById(R.id.buttonPanel);
        ViewGroup d2 = alertController.d(findViewById6, findViewById3);
        ViewGroup d3 = alertController.d(findViewById7, findViewById4);
        ViewGroup d4 = alertController.d(findViewById8, findViewById5);
        NestedScrollView nestedScrollView = (NestedScrollView) alertController.f61c.findViewById(R.id.scrollView);
        alertController.A = nestedScrollView;
        nestedScrollView.setFocusable(false);
        alertController.A.setNestedScrollingEnabled(false);
        TextView textView = (TextView) d3.findViewById(16908299);
        alertController.F = textView;
        if (textView != null) {
            CharSequence charSequence = alertController.f64f;
            if (charSequence != null) {
                textView.setText(charSequence);
            } else {
                textView.setVisibility(8);
                alertController.A.removeView(alertController.F);
                if (alertController.f65g != null) {
                    ViewGroup viewGroup2 = (ViewGroup) alertController.A.getParent();
                    int indexOfChild = viewGroup2.indexOfChild(alertController.A);
                    viewGroup2.removeViewAt(indexOfChild);
                    viewGroup2.addView(alertController.f65g, indexOfChild, new ViewGroup.LayoutParams(-1, -1));
                } else {
                    d3.setVisibility(8);
                }
            }
        }
        Button button = (Button) d4.findViewById(16908313);
        alertController.o = button;
        button.setOnClickListener(alertController.R);
        if (TextUtils.isEmpty(alertController.p) && alertController.r == null) {
            alertController.o.setVisibility(8);
            z = false;
        } else {
            alertController.o.setText(alertController.p);
            Drawable drawable = alertController.r;
            if (drawable != null) {
                int i2 = alertController.f62d;
                drawable.setBounds(0, 0, i2, i2);
                alertController.o.setCompoundDrawables(alertController.r, null, null, null);
            }
            alertController.o.setVisibility(0);
            z = true;
        }
        Button button2 = (Button) d4.findViewById(16908314);
        alertController.s = button2;
        button2.setOnClickListener(alertController.R);
        if (TextUtils.isEmpty(alertController.t) && alertController.v == null) {
            alertController.s.setVisibility(8);
        } else {
            alertController.s.setText(alertController.t);
            Drawable drawable2 = alertController.v;
            if (drawable2 != null) {
                int i3 = alertController.f62d;
                drawable2.setBounds(0, 0, i3, i3);
                alertController.s.setCompoundDrawables(alertController.v, null, null, null);
            }
            alertController.s.setVisibility(0);
            z |= true;
        }
        Button button3 = (Button) d4.findViewById(16908315);
        alertController.w = button3;
        button3.setOnClickListener(alertController.R);
        if (TextUtils.isEmpty(alertController.x) && alertController.z == null) {
            alertController.w.setVisibility(8);
            view = null;
        } else {
            alertController.w.setText(alertController.x);
            Drawable drawable3 = alertController.z;
            if (drawable3 != null) {
                int i4 = alertController.f62d;
                drawable3.setBounds(0, 0, i4, i4);
                view = null;
                alertController.w.setCompoundDrawables(alertController.z, null, null, null);
            } else {
                view = null;
            }
            alertController.w.setVisibility(0);
            z |= true;
        }
        Context context = alertController.f59a;
        TypedValue typedValue = new TypedValue();
        context.getTheme().resolveAttribute(R.attr.alertDialogCenterButtons, typedValue, true);
        if (typedValue.data != 0) {
            if (z) {
                alertController.b(alertController.o);
            } else if (z) {
                alertController.b(alertController.s);
            } else if (z) {
                alertController.b(alertController.w);
            }
        }
        if (!(z)) {
            d4.setVisibility(8);
        }
        if (alertController.G != null) {
            d2.addView(alertController.G, 0, new ViewGroup.LayoutParams(-1, -2));
            alertController.f61c.findViewById(R.id.title_template).setVisibility(8);
        } else {
            alertController.D = (ImageView) alertController.f61c.findViewById(16908294);
            if ((!TextUtils.isEmpty(alertController.f63e)) && alertController.P) {
                TextView textView2 = (TextView) alertController.f61c.findViewById(R.id.alertTitle);
                alertController.E = textView2;
                textView2.setText(alertController.f63e);
                int i5 = alertController.B;
                if (i5 != 0) {
                    alertController.D.setImageResource(i5);
                } else {
                    Drawable drawable4 = alertController.C;
                    if (drawable4 != null) {
                        alertController.D.setImageDrawable(drawable4);
                    } else {
                        alertController.E.setPadding(alertController.D.getPaddingLeft(), alertController.D.getPaddingTop(), alertController.D.getPaddingRight(), alertController.D.getPaddingBottom());
                        alertController.D.setVisibility(8);
                    }
                }
            } else {
                alertController.f61c.findViewById(R.id.title_template).setVisibility(8);
                alertController.D.setVisibility(8);
                d2.setVisibility(8);
            }
        }
        boolean z3 = viewGroup.getVisibility() != 8;
        int i6 = (d2 == null || d2.getVisibility() == 8) ? 0 : 1;
        boolean z4 = d4.getVisibility() != 8;
        if (!z4 && (findViewById = d3.findViewById(R.id.textSpacerNoButtons)) != null) {
            findViewById.setVisibility(0);
        }
        if (i6 != 0) {
            NestedScrollView nestedScrollView2 = alertController.A;
            if (nestedScrollView2 != null) {
                nestedScrollView2.setClipToPadding(true);
            }
            View findViewById9 = (alertController.f64f == null && alertController.f65g == null) ? view : d2.findViewById(R.id.titleDividerNoCustom);
            if (findViewById9 != null) {
                findViewById9.setVisibility(0);
            }
        } else {
            View findViewById10 = d3.findViewById(R.id.textSpacerNoTitle);
            if (findViewById10 != null) {
                findViewById10.setVisibility(0);
            }
        }
        ListView listView = alertController.f65g;
        if (listView instanceof AlertController.RecycleListView) {
            AlertController.RecycleListView recycleListView = (AlertController.RecycleListView) listView;
            Objects.requireNonNull(recycleListView);
            if (!z4 || i6 == 0) {
                recycleListView.setPadding(recycleListView.getPaddingLeft(), i6 != 0 ? recycleListView.getPaddingTop() : recycleListView.f67b, recycleListView.getPaddingRight(), z4 ? recycleListView.getPaddingBottom() : recycleListView.f68c);
            }
        }
        if (!z3) {
            View view3 = alertController.f65g;
            if (view3 == null) {
                view3 = alertController.A;
            }
            if (view3 != null) {
                int i7 = z4 ? 2 : 0;
                View findViewById11 = alertController.f61c.findViewById(R.id.scrollIndicatorUp);
                View findViewById12 = alertController.f61c.findViewById(R.id.scrollIndicatorDown);
                AtomicInteger atomicInteger = b.j.j.q.f2214a;
                view3.setScrollIndicators(i6 | i7, 3);
                if (findViewById11 != null) {
                    d3.removeView(findViewById11);
                }
                if (findViewById12 != null) {
                    d3.removeView(findViewById12);
                }
            }
        }
        ListView listView2 = alertController.f65g;
        if (listView2 == null || (listAdapter = alertController.H) == null) {
            return;
        }
        listView2.setAdapter(listAdapter);
        int i8 = alertController.I;
        if (i8 > -1) {
            listView2.setItemChecked(i8, true);
            listView2.setSelection(i8);
        }
    }

    @Override // android.app.Dialog, android.view.KeyEvent.Callback
    public boolean onKeyDown(int i, KeyEvent keyEvent) {
        NestedScrollView nestedScrollView = this.f564b.A;
        if (nestedScrollView != null && nestedScrollView.i(keyEvent)) {
            return true;
        }
        return super.onKeyDown(i, keyEvent);
    }

    @Override // android.app.Dialog, android.view.KeyEvent.Callback
    public boolean onKeyUp(int i, KeyEvent keyEvent) {
        NestedScrollView nestedScrollView = this.f564b.A;
        if (nestedScrollView != null && nestedScrollView.i(keyEvent)) {
            return true;
        }
        return super.onKeyUp(i, keyEvent);
    }

    @Override // b.b.c.p, android.app.Dialog
    public void setTitle(CharSequence charSequence) {
        super.setTitle(charSequence);
        AlertController alertController = this.f564b;
        alertController.f63e = charSequence;
        TextView textView = alertController.E;
        if (textView != null) {
            textView.setText(charSequence);
        }
    }
}