package androidx.appcompat.widget;

import android.annotation.SuppressLint;
import android.app.PendingIntent;
import android.app.SearchableInfo;
import android.content.ActivityNotFoundException;
import android.content.ComponentName;
import android.content.Context;
import android.content.Intent;
import android.content.res.Configuration;
import android.content.res.Resources;
import android.content.res.TypedArray;
import android.database.Cursor;
import android.graphics.Rect;
import android.graphics.drawable.Drawable;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Parcel;
import android.os.Parcelable;
import android.text.Editable;
import android.text.SpannableStringBuilder;
import android.text.TextUtils;
import android.text.TextWatcher;
import android.text.style.ImageSpan;
import android.util.AttributeSet;
import android.util.Log;
import android.util.TypedValue;
import android.view.KeyEvent;
import android.view.LayoutInflater;
import android.view.MotionEvent;
import android.view.TouchDelegate;
import android.view.View;
import android.view.ViewConfiguration;
import android.view.ViewGroup;
import android.view.inputmethod.EditorInfo;
import android.view.inputmethod.InputConnection;
import android.view.inputmethod.InputMethodManager;
import android.widget.AdapterView;
import android.widget.AutoCompleteTextView;
import android.widget.ImageView;
import android.widget.TextView;
import b.b.h.e1;
import b.b.h.i0;
import b.b.h.r0;
import b.b.h.y0;
import b.j.j.q;
import com.google.android.gms.actions.SearchIntents;
import com.ibosoninnov.unitear.R;
import java.lang.reflect.Method;
import java.util.Objects;
import java.util.WeakHashMap;
import java.util.concurrent.atomic.AtomicInteger;

/* loaded from: classes.dex */
public class SearchView extends i0 implements b.b.g.b {

    /* renamed from: b  reason: collision with root package name */
    public static final n f139b;
    public m A;
    public View.OnClickListener B;
    public boolean C;
    public boolean D;
    public b.k.a.a E;
    public boolean F;
    public CharSequence G;
    public boolean H;
    public boolean I;
    public int J;
    public boolean K;
    public CharSequence L;
    public CharSequence M;
    public boolean N;
    public int O;
    public SearchableInfo P;
    public Bundle Q;
    public final Runnable R;
    public Runnable S;
    public final WeakHashMap<String, Drawable.ConstantState> T;
    public final View.OnClickListener U;
    public View.OnKeyListener V;
    public final TextView.OnEditorActionListener W;
    public final AdapterView.OnItemClickListener a0;
    public final AdapterView.OnItemSelectedListener b0;

    /* renamed from: c  reason: collision with root package name */
    public final SearchAutoComplete f140c;
    public TextWatcher c0;

    /* renamed from: d  reason: collision with root package name */
    public final View f141d;

    /* renamed from: e  reason: collision with root package name */
    public final View f142e;

    /* renamed from: f  reason: collision with root package name */
    public final View f143f;

    /* renamed from: g  reason: collision with root package name */
    public final ImageView f144g;

    /* renamed from: h  reason: collision with root package name */
    public final ImageView f145h;
    public final ImageView i;
    public final ImageView j;
    public final View k;
    public p l;
    public Rect m;
    public Rect n;
    public int[] o;
    public int[] p;
    public final ImageView q;
    public final Drawable r;
    public final int s;
    public final int t;
    public final Intent u;
    public final Intent v;
    public final CharSequence w;
    public l x;
    public k y;
    public View.OnFocusChangeListener z;

    /* loaded from: classes.dex */
    public static class SearchAutoComplete extends b.b.h.d {

        /* renamed from: b  reason: collision with root package name */
        public int f146b;

        /* renamed from: c  reason: collision with root package name */
        public SearchView f147c;

        /* renamed from: d  reason: collision with root package name */
        public boolean f148d;

        /* renamed from: e  reason: collision with root package name */
        public final Runnable f149e;

        /* loaded from: classes.dex */
        public class a implements Runnable {
            public a() {
            }

            @Override // java.lang.Runnable
            public void run() {
                SearchAutoComplete searchAutoComplete = SearchAutoComplete.this;
                if (searchAutoComplete.f148d) {
                    ((InputMethodManager) searchAutoComplete.getContext().getSystemService("input_method")).showSoftInput(searchAutoComplete, 0);
                    searchAutoComplete.f148d = false;
                }
            }
        }

        public SearchAutoComplete(Context context, AttributeSet attributeSet) {
            super(context, attributeSet, R.attr.autoCompleteTextViewStyle);
            this.f149e = new a();
            this.f146b = getThreshold();
        }

        private int getSearchViewTextMinWidthDp() {
            Configuration configuration = getResources().getConfiguration();
            int i = configuration.screenWidthDp;
            int i2 = configuration.screenHeightDp;
            if (i < 960 || i2 < 720 || configuration.orientation != 2) {
                if (i < 600) {
                    return (i < 640 || i2 < 480) ? 160 : 192;
                }
                return 192;
            }
            return 256;
        }

        public void a() {
            if (Build.VERSION.SDK_INT >= 29) {
                setInputMethodMode(1);
                if (enoughToFilter()) {
                    showDropDown();
                    return;
                }
                return;
            }
            n nVar = SearchView.f139b;
            Objects.requireNonNull(nVar);
            n.a();
            Method method = nVar.f163c;
            if (method != null) {
                try {
                    method.invoke(this, Boolean.TRUE);
                } catch (Exception unused) {
                }
            }
        }

        @Override // android.widget.AutoCompleteTextView
        public boolean enoughToFilter() {
            return this.f146b <= 0 || super.enoughToFilter();
        }

        @Override // b.b.h.d, android.widget.TextView, android.view.View
        public InputConnection onCreateInputConnection(EditorInfo editorInfo) {
            InputConnection onCreateInputConnection = super.onCreateInputConnection(editorInfo);
            if (this.f148d) {
                removeCallbacks(this.f149e);
                post(this.f149e);
            }
            return onCreateInputConnection;
        }

        @Override // android.view.View
        public void onFinishInflate() {
            super.onFinishInflate();
            setMinWidth((int) TypedValue.applyDimension(1, getSearchViewTextMinWidthDp(), getResources().getDisplayMetrics()));
        }

        @Override // android.widget.AutoCompleteTextView, android.widget.TextView, android.view.View
        public void onFocusChanged(boolean z, int i, Rect rect) {
            super.onFocusChanged(z, i, rect);
            SearchView searchView = this.f147c;
            searchView.t(searchView.D);
            searchView.post(searchView.R);
            if (searchView.f140c.hasFocus()) {
                searchView.f();
            }
        }

        @Override // android.widget.AutoCompleteTextView, android.widget.TextView, android.view.View
        public boolean onKeyPreIme(int i, KeyEvent keyEvent) {
            if (i == 4) {
                if (keyEvent.getAction() == 0 && keyEvent.getRepeatCount() == 0) {
                    KeyEvent.DispatcherState keyDispatcherState = getKeyDispatcherState();
                    if (keyDispatcherState != null) {
                        keyDispatcherState.startTracking(keyEvent, this);
                    }
                    return true;
                } else if (keyEvent.getAction() == 1) {
                    KeyEvent.DispatcherState keyDispatcherState2 = getKeyDispatcherState();
                    if (keyDispatcherState2 != null) {
                        keyDispatcherState2.handleUpEvent(keyEvent);
                    }
                    if (keyEvent.isTracking() && !keyEvent.isCanceled()) {
                        this.f147c.clearFocus();
                        setImeVisibility(false);
                        return true;
                    }
                }
            }
            return super.onKeyPreIme(i, keyEvent);
        }

        @Override // android.widget.AutoCompleteTextView, android.widget.TextView, android.view.View
        public void onWindowFocusChanged(boolean z) {
            super.onWindowFocusChanged(z);
            if (z && this.f147c.hasFocus() && getVisibility() == 0) {
                this.f148d = true;
                Context context = getContext();
                n nVar = SearchView.f139b;
                if (context.getResources().getConfiguration().orientation == 2) {
                    a();
                }
            }
        }

        @Override // android.widget.AutoCompleteTextView
        public void performCompletion() {
        }

        @Override // android.widget.AutoCompleteTextView
        public void replaceText(CharSequence charSequence) {
        }

        public void setImeVisibility(boolean z) {
            InputMethodManager inputMethodManager = (InputMethodManager) getContext().getSystemService("input_method");
            if (!z) {
                this.f148d = false;
                removeCallbacks(this.f149e);
                inputMethodManager.hideSoftInputFromWindow(getWindowToken(), 0);
            } else if (inputMethodManager.isActive(this)) {
                this.f148d = false;
                removeCallbacks(this.f149e);
                inputMethodManager.showSoftInput(this, 0);
            } else {
                this.f148d = true;
            }
        }

        public void setSearchView(SearchView searchView) {
            this.f147c = searchView;
        }

        @Override // android.widget.AutoCompleteTextView
        public void setThreshold(int i) {
            super.setThreshold(i);
            this.f146b = i;
        }
    }

    /* loaded from: classes.dex */
    public class a implements TextWatcher {
        public a() {
        }

        @Override // android.text.TextWatcher
        public void afterTextChanged(Editable editable) {
        }

        @Override // android.text.TextWatcher
        public void beforeTextChanged(CharSequence charSequence, int i, int i2, int i3) {
        }

        @Override // android.text.TextWatcher
        public void onTextChanged(CharSequence charSequence, int i, int i2, int i3) {
            SearchView searchView = SearchView.this;
            Editable text = searchView.f140c.getText();
            searchView.M = text;
            boolean z = !TextUtils.isEmpty(text);
            searchView.s(z);
            searchView.u(!z);
            searchView.o();
            searchView.r();
            if (searchView.x != null && !TextUtils.equals(charSequence, searchView.L)) {
                searchView.x.e(charSequence.toString());
            }
            searchView.L = charSequence.toString();
        }
    }

    /* loaded from: classes.dex */
    public class b implements Runnable {
        public b() {
        }

        @Override // java.lang.Runnable
        public void run() {
            SearchView.this.p();
        }
    }

    /* loaded from: classes.dex */
    public class c implements Runnable {
        public c() {
        }

        @Override // java.lang.Runnable
        public void run() {
            b.k.a.a aVar = SearchView.this.E;
            if (aVar instanceof r0) {
                aVar.b(null);
            }
        }
    }

    /* loaded from: classes.dex */
    public class d implements View.OnFocusChangeListener {
        public d() {
        }

        @Override // android.view.View.OnFocusChangeListener
        public void onFocusChange(View view, boolean z) {
            SearchView searchView = SearchView.this;
            View.OnFocusChangeListener onFocusChangeListener = searchView.z;
            if (onFocusChangeListener != null) {
                onFocusChangeListener.onFocusChange(searchView, z);
            }
        }
    }

    /* loaded from: classes.dex */
    public class e implements View.OnLayoutChangeListener {
        public e() {
        }

        @Override // android.view.View.OnLayoutChangeListener
        public void onLayoutChange(View view, int i, int i2, int i3, int i4, int i5, int i6, int i7, int i8) {
            int i9;
            int i10;
            SearchView searchView = SearchView.this;
            if (searchView.k.getWidth() > 1) {
                Resources resources = searchView.getContext().getResources();
                int paddingLeft = searchView.f142e.getPaddingLeft();
                Rect rect = new Rect();
                boolean b2 = e1.b(searchView);
                if (searchView.C) {
                    i9 = resources.getDimensionPixelSize(R.dimen.abc_dropdownitem_text_padding_left) + resources.getDimensionPixelSize(R.dimen.abc_dropdownitem_icon_width);
                } else {
                    i9 = 0;
                }
                searchView.f140c.getDropDownBackground().getPadding(rect);
                if (b2) {
                    i10 = -rect.left;
                } else {
                    i10 = paddingLeft - (rect.left + i9);
                }
                searchView.f140c.setDropDownHorizontalOffset(i10);
                searchView.f140c.setDropDownWidth((((searchView.k.getWidth() + rect.left) + rect.right) + i9) - paddingLeft);
            }
        }
    }

    /* loaded from: classes.dex */
    public class f implements View.OnClickListener {
        public f() {
        }

        @Override // android.view.View.OnClickListener
        public void onClick(View view) {
            SearchView searchView = SearchView.this;
            if (view == searchView.f144g) {
                searchView.l();
            } else if (view == searchView.i) {
                searchView.h();
            } else if (view == searchView.f145h) {
                searchView.m();
            } else if (view == searchView.j) {
                SearchableInfo searchableInfo = searchView.P;
                if (searchableInfo == null) {
                    return;
                }
                try {
                    if (searchableInfo.getVoiceSearchLaunchWebSearch()) {
                        Intent intent = new Intent(searchView.u);
                        ComponentName searchActivity = searchableInfo.getSearchActivity();
                        intent.putExtra("calling_package", searchActivity == null ? null : searchActivity.flattenToShortString());
                        searchView.getContext().startActivity(intent);
                    } else if (searchableInfo.getVoiceSearchLaunchRecognizer()) {
                        searchView.getContext().startActivity(searchView.e(searchView.v, searchableInfo));
                    }
                } catch (ActivityNotFoundException unused) {
                    Log.w("SearchView", "Could not find voice search activity");
                }
            } else if (view == searchView.f140c) {
                searchView.f();
            }
        }
    }

    /* loaded from: classes.dex */
    public class g implements View.OnKeyListener {
        public g() {
        }

        @Override // android.view.View.OnKeyListener
        public boolean onKey(View view, int i, KeyEvent keyEvent) {
            SearchView searchView = SearchView.this;
            if (searchView.P == null) {
                return false;
            }
            if (searchView.f140c.isPopupShowing() && SearchView.this.f140c.getListSelection() != -1) {
                return SearchView.this.n(i, keyEvent);
            }
            if (!(TextUtils.getTrimmedLength(SearchView.this.f140c.getText()) == 0) && keyEvent.hasNoModifiers() && keyEvent.getAction() == 1 && i == 66) {
                view.cancelLongPress();
                SearchView searchView2 = SearchView.this;
                searchView2.g(0, null, searchView2.f140c.getText().toString());
                return true;
            }
            return false;
        }
    }

    /* loaded from: classes.dex */
    public class h implements TextView.OnEditorActionListener {
        public h() {
        }

        @Override // android.widget.TextView.OnEditorActionListener
        public boolean onEditorAction(TextView textView, int i, KeyEvent keyEvent) {
            SearchView.this.m();
            return true;
        }
    }

    /* loaded from: classes.dex */
    public class i implements AdapterView.OnItemClickListener {
        public i() {
        }

        @Override // android.widget.AdapterView.OnItemClickListener
        public void onItemClick(AdapterView<?> adapterView, View view, int i, long j) {
            SearchView.this.i(i);
        }
    }

    /* loaded from: classes.dex */
    public class j implements AdapterView.OnItemSelectedListener {
        public j() {
        }

        @Override // android.widget.AdapterView.OnItemSelectedListener
        public void onItemSelected(AdapterView<?> adapterView, View view, int i, long j) {
            SearchView.this.j(i);
        }

        @Override // android.widget.AdapterView.OnItemSelectedListener
        public void onNothingSelected(AdapterView<?> adapterView) {
        }
    }

    /* loaded from: classes.dex */
    public interface k {
        boolean a();
    }

    /* loaded from: classes.dex */
    public interface l {
        boolean e(String str);

        boolean g(String str);
    }

    /* loaded from: classes.dex */
    public interface m {
        boolean a(int i);

        boolean b(int i);
    }

    /* loaded from: classes.dex */
    public static class n {

        /* renamed from: a  reason: collision with root package name */
        public Method f161a;

        /* renamed from: b  reason: collision with root package name */
        public Method f162b;

        /* renamed from: c  reason: collision with root package name */
        public Method f163c;

        @SuppressLint({"DiscouragedPrivateApi", "SoonBlockedPrivateApi"})
        public n() {
            this.f161a = null;
            this.f162b = null;
            this.f163c = null;
            a();
            try {
                Method declaredMethod = AutoCompleteTextView.class.getDeclaredMethod("doBeforeTextChanged", new Class[0]);
                this.f161a = declaredMethod;
                declaredMethod.setAccessible(true);
            } catch (NoSuchMethodException unused) {
            }
            try {
                Method declaredMethod2 = AutoCompleteTextView.class.getDeclaredMethod("doAfterTextChanged", new Class[0]);
                this.f162b = declaredMethod2;
                declaredMethod2.setAccessible(true);
            } catch (NoSuchMethodException unused2) {
            }
            try {
                Method method = AutoCompleteTextView.class.getMethod("ensureImeVisible", Boolean.TYPE);
                this.f163c = method;
                method.setAccessible(true);
            } catch (NoSuchMethodException unused3) {
            }
        }

        public static void a() {
            if (Build.VERSION.SDK_INT >= 29) {
                throw new UnsupportedClassVersionError("This function can only be used for API Level < 29.");
            }
        }
    }

    /* loaded from: classes.dex */
    public static class o extends b.l.a.a {
        public static final Parcelable.Creator<o> CREATOR = new a();

        /* renamed from: b  reason: collision with root package name */
        public boolean f164b;

        /* loaded from: classes.dex */
        public class a implements Parcelable.ClassLoaderCreator<o> {
            /* JADX DEBUG: Return type fixed from 'java.lang.Object' to match base method */
            @Override // android.os.Parcelable.ClassLoaderCreator
            public o createFromParcel(Parcel parcel, ClassLoader classLoader) {
                return new o(parcel, classLoader);
            }

            @Override // android.os.Parcelable.Creator
            public Object[] newArray(int i) {
                return new o[i];
            }

            @Override // android.os.Parcelable.Creator
            public Object createFromParcel(Parcel parcel) {
                return new o(parcel, null);
            }
        }

        public o(Parcelable parcelable) {
            super(parcelable);
        }

        public String toString() {
            StringBuilder x = c.b.a.a.a.x("SearchView.SavedState{");
            x.append(Integer.toHexString(System.identityHashCode(this)));
            x.append(" isIconified=");
            x.append(this.f164b);
            x.append("}");
            return x.toString();
        }

        @Override // b.l.a.a, android.os.Parcelable
        public void writeToParcel(Parcel parcel, int i) {
            super.writeToParcel(parcel, i);
            parcel.writeValue(Boolean.valueOf(this.f164b));
        }

        public o(Parcel parcel, ClassLoader classLoader) {
            super(parcel, classLoader);
            this.f164b = ((Boolean) parcel.readValue(null)).booleanValue();
        }
    }

    /* loaded from: classes.dex */
    public static class p extends TouchDelegate {

        /* renamed from: a  reason: collision with root package name */
        public final View f165a;

        /* renamed from: b  reason: collision with root package name */
        public final Rect f166b;

        /* renamed from: c  reason: collision with root package name */
        public final Rect f167c;

        /* renamed from: d  reason: collision with root package name */
        public final Rect f168d;

        /* renamed from: e  reason: collision with root package name */
        public final int f169e;

        /* renamed from: f  reason: collision with root package name */
        public boolean f170f;

        public p(Rect rect, Rect rect2, View view) {
            super(rect, view);
            this.f169e = ViewConfiguration.get(view.getContext()).getScaledTouchSlop();
            this.f166b = new Rect();
            this.f168d = new Rect();
            this.f167c = new Rect();
            a(rect, rect2);
            this.f165a = view;
        }

        public void a(Rect rect, Rect rect2) {
            this.f166b.set(rect);
            this.f168d.set(rect);
            Rect rect3 = this.f168d;
            int i = this.f169e;
            rect3.inset(-i, -i);
            this.f167c.set(rect2);
        }

        @Override // android.view.TouchDelegate
        public boolean onTouchEvent(MotionEvent motionEvent) {
            boolean z;
            boolean z2;
            int x = (int) motionEvent.getX();
            int y = (int) motionEvent.getY();
            int action = motionEvent.getAction();
            boolean z3 = true;
            if (action != 0) {
                if (action == 1 || action == 2) {
                    z2 = this.f170f;
                    if (z2 && !this.f168d.contains(x, y)) {
                        z3 = z2;
                        z = false;
                    }
                } else {
                    if (action == 3) {
                        z2 = this.f170f;
                        this.f170f = false;
                    }
                    z = true;
                    z3 = false;
                }
                z3 = z2;
                z = true;
            } else {
                if (this.f166b.contains(x, y)) {
                    this.f170f = true;
                    z = true;
                }
                z = true;
                z3 = false;
            }
            if (z3) {
                if (z && !this.f167c.contains(x, y)) {
                    motionEvent.setLocation(this.f165a.getWidth() / 2, this.f165a.getHeight() / 2);
                } else {
                    Rect rect = this.f167c;
                    motionEvent.setLocation(x - rect.left, y - rect.top);
                }
                return this.f165a.dispatchTouchEvent(motionEvent);
            }
            return false;
        }
    }

    static {
        f139b = Build.VERSION.SDK_INT < 29 ? new n() : null;
    }

    public SearchView(Context context) {
        this(context, null);
    }

    private int getPreferredHeight() {
        return getContext().getResources().getDimensionPixelSize(R.dimen.abc_search_view_preferred_height);
    }

    private int getPreferredWidth() {
        return getContext().getResources().getDimensionPixelSize(R.dimen.abc_search_view_preferred_width);
    }

    private void setQuery(CharSequence charSequence) {
        this.f140c.setText(charSequence);
        this.f140c.setSelection(TextUtils.isEmpty(charSequence) ? 0 : charSequence.length());
    }

    @Override // b.b.g.b
    public void b() {
        if (this.N) {
            return;
        }
        this.N = true;
        int imeOptions = this.f140c.getImeOptions();
        this.O = imeOptions;
        this.f140c.setImeOptions(imeOptions | 33554432);
        this.f140c.setText("");
        setIconified(false);
    }

    @Override // b.b.g.b
    public void c() {
        this.f140c.setText("");
        SearchAutoComplete searchAutoComplete = this.f140c;
        searchAutoComplete.setSelection(searchAutoComplete.length());
        this.M = "";
        clearFocus();
        t(true);
        this.f140c.setImeOptions(this.O);
        this.N = false;
    }

    @Override // android.view.ViewGroup, android.view.View
    public void clearFocus() {
        this.I = true;
        super.clearFocus();
        this.f140c.clearFocus();
        this.f140c.setImeVisibility(false);
        this.I = false;
    }

    public final Intent d(String str, Uri uri, String str2, String str3, int i2, String str4) {
        Intent intent = new Intent(str);
        intent.addFlags(268435456);
        if (uri != null) {
            intent.setData(uri);
        }
        intent.putExtra("user_query", this.M);
        if (str3 != null) {
            intent.putExtra(SearchIntents.EXTRA_QUERY, str3);
        }
        if (str2 != null) {
            intent.putExtra("intent_extra_data_key", str2);
        }
        Bundle bundle = this.Q;
        if (bundle != null) {
            intent.putExtra("app_data", bundle);
        }
        if (i2 != 0) {
            intent.putExtra("action_key", i2);
            intent.putExtra("action_msg", str4);
        }
        intent.setComponent(this.P.getSearchActivity());
        return intent;
    }

    public final Intent e(Intent intent, SearchableInfo searchableInfo) {
        ComponentName searchActivity = searchableInfo.getSearchActivity();
        Intent intent2 = new Intent("android.intent.action.SEARCH");
        intent2.setComponent(searchActivity);
        PendingIntent activity = PendingIntent.getActivity(getContext(), 0, intent2, 1073741824);
        Bundle bundle = new Bundle();
        Bundle bundle2 = this.Q;
        if (bundle2 != null) {
            bundle.putParcelable("app_data", bundle2);
        }
        Intent intent3 = new Intent(intent);
        Resources resources = getResources();
        String string = searchableInfo.getVoiceLanguageModeId() != 0 ? resources.getString(searchableInfo.getVoiceLanguageModeId()) : "free_form";
        String string2 = searchableInfo.getVoicePromptTextId() != 0 ? resources.getString(searchableInfo.getVoicePromptTextId()) : null;
        String string3 = searchableInfo.getVoiceLanguageId() != 0 ? resources.getString(searchableInfo.getVoiceLanguageId()) : null;
        int voiceMaxResults = searchableInfo.getVoiceMaxResults() != 0 ? searchableInfo.getVoiceMaxResults() : 1;
        intent3.putExtra("android.speech.extra.LANGUAGE_MODEL", string);
        intent3.putExtra("android.speech.extra.PROMPT", string2);
        intent3.putExtra("android.speech.extra.LANGUAGE", string3);
        intent3.putExtra("android.speech.extra.MAX_RESULTS", voiceMaxResults);
        intent3.putExtra("calling_package", searchActivity != null ? searchActivity.flattenToShortString() : null);
        intent3.putExtra("android.speech.extra.RESULTS_PENDINGINTENT", activity);
        intent3.putExtra("android.speech.extra.RESULTS_PENDINGINTENT_BUNDLE", bundle);
        return intent3;
    }

    public void f() {
        if (Build.VERSION.SDK_INT >= 29) {
            this.f140c.refreshAutoCompleteResults();
            return;
        }
        n nVar = f139b;
        SearchAutoComplete searchAutoComplete = this.f140c;
        Objects.requireNonNull(nVar);
        n.a();
        Method method = nVar.f161a;
        if (method != null) {
            try {
                method.invoke(searchAutoComplete, new Object[0]);
            } catch (Exception unused) {
            }
        }
        n nVar2 = f139b;
        SearchAutoComplete searchAutoComplete2 = this.f140c;
        Objects.requireNonNull(nVar2);
        n.a();
        Method method2 = nVar2.f162b;
        if (method2 != null) {
            try {
                method2.invoke(searchAutoComplete2, new Object[0]);
            } catch (Exception unused2) {
            }
        }
    }

    public void g(int i2, String str, String str2) {
        getContext().startActivity(d("android.intent.action.SEARCH", null, null, str2, i2, null));
    }

    public int getImeOptions() {
        return this.f140c.getImeOptions();
    }

    public int getInputType() {
        return this.f140c.getInputType();
    }

    public int getMaxWidth() {
        return this.J;
    }

    public CharSequence getQuery() {
        return this.f140c.getText();
    }

    public CharSequence getQueryHint() {
        CharSequence charSequence = this.G;
        if (charSequence != null) {
            return charSequence;
        }
        SearchableInfo searchableInfo = this.P;
        if (searchableInfo != null && searchableInfo.getHintId() != 0) {
            return getContext().getText(this.P.getHintId());
        }
        return this.w;
    }

    public int getSuggestionCommitIconResId() {
        return this.t;
    }

    public int getSuggestionRowLayout() {
        return this.s;
    }

    public b.k.a.a getSuggestionsAdapter() {
        return this.E;
    }

    public void h() {
        if (TextUtils.isEmpty(this.f140c.getText())) {
            if (this.C) {
                k kVar = this.y;
                if (kVar == null || !kVar.a()) {
                    clearFocus();
                    t(true);
                    return;
                }
                return;
            }
            return;
        }
        this.f140c.setText("");
        this.f140c.requestFocus();
        this.f140c.setImeVisibility(true);
    }

    public boolean i(int i2) {
        int i3;
        String h2;
        m mVar = this.A;
        if (mVar == null || !mVar.b(i2)) {
            Cursor cursor = this.E.f2306d;
            if (cursor != null && cursor.moveToPosition(i2)) {
                Intent intent = null;
                try {
                    int i4 = r0.m;
                    String h3 = r0.h(cursor, cursor.getColumnIndex("suggest_intent_action"));
                    if (h3 == null) {
                        h3 = this.P.getSuggestIntentAction();
                    }
                    if (h3 == null) {
                        h3 = "android.intent.action.SEARCH";
                    }
                    String str = h3;
                    String h4 = r0.h(cursor, cursor.getColumnIndex("suggest_intent_data"));
                    if (h4 == null) {
                        h4 = this.P.getSuggestIntentData();
                    }
                    if (h4 != null && (h2 = r0.h(cursor, cursor.getColumnIndex("suggest_intent_data_id"))) != null) {
                        h4 = h4 + "/" + Uri.encode(h2);
                    }
                    intent = d(str, h4 == null ? null : Uri.parse(h4), r0.h(cursor, cursor.getColumnIndex("suggest_intent_extra_data")), r0.h(cursor, cursor.getColumnIndex("suggest_intent_query")), 0, null);
                } catch (RuntimeException e2) {
                    try {
                        i3 = cursor.getPosition();
                    } catch (RuntimeException unused) {
                        i3 = -1;
                    }
                    Log.w("SearchView", "Search suggestions cursor at row " + i3 + " returned exception.", e2);
                }
                if (intent != null) {
                    try {
                        getContext().startActivity(intent);
                    } catch (RuntimeException e3) {
                        Log.e("SearchView", "Failed launch activity: " + intent, e3);
                    }
                }
            }
            this.f140c.setImeVisibility(false);
            this.f140c.dismissDropDown();
            return true;
        }
        return false;
    }

    public boolean j(int i2) {
        m mVar = this.A;
        if (mVar == null || !mVar.a(i2)) {
            Editable text = this.f140c.getText();
            Cursor cursor = this.E.f2306d;
            if (cursor == null) {
                return true;
            }
            if (cursor.moveToPosition(i2)) {
                CharSequence c2 = this.E.c(cursor);
                if (c2 != null) {
                    setQuery(c2);
                    return true;
                }
                setQuery(text);
                return true;
            }
            setQuery(text);
            return true;
        }
        return false;
    }

    public void k(CharSequence charSequence) {
        setQuery(charSequence);
    }

    public void l() {
        t(false);
        this.f140c.requestFocus();
        this.f140c.setImeVisibility(true);
        View.OnClickListener onClickListener = this.B;
        if (onClickListener != null) {
            onClickListener.onClick(this);
        }
    }

    public void m() {
        Editable text = this.f140c.getText();
        if (text == null || TextUtils.getTrimmedLength(text) <= 0) {
            return;
        }
        l lVar = this.x;
        if (lVar == null || !lVar.g(text.toString())) {
            if (this.P != null) {
                g(0, null, text.toString());
            }
            this.f140c.setImeVisibility(false);
            this.f140c.dismissDropDown();
        }
    }

    public boolean n(int i2, KeyEvent keyEvent) {
        if (this.P != null && this.E != null && keyEvent.getAction() == 0 && keyEvent.hasNoModifiers()) {
            if (i2 == 66 || i2 == 84 || i2 == 61) {
                return i(this.f140c.getListSelection());
            }
            if (i2 != 21 && i2 != 22) {
                if (i2 != 19 || this.f140c.getListSelection() == 0) {
                    return false;
                }
            } else {
                this.f140c.setSelection(i2 == 21 ? 0 : this.f140c.length());
                this.f140c.setListSelection(0);
                this.f140c.clearListSelection();
                this.f140c.a();
                return true;
            }
        }
        return false;
    }

    public final void o() {
        boolean z = true;
        boolean z2 = !TextUtils.isEmpty(this.f140c.getText());
        if (!z2 && (!this.C || this.N)) {
            z = false;
        }
        this.i.setVisibility(z ? 0 : 8);
        Drawable drawable = this.i.getDrawable();
        if (drawable != null) {
            drawable.setState(z2 ? ViewGroup.ENABLED_STATE_SET : ViewGroup.EMPTY_STATE_SET);
        }
    }

    @Override // android.view.ViewGroup, android.view.View
    public void onDetachedFromWindow() {
        removeCallbacks(this.R);
        post(this.S);
        super.onDetachedFromWindow();
    }

    @Override // b.b.h.i0, android.view.ViewGroup, android.view.View
    public void onLayout(boolean z, int i2, int i3, int i4, int i5) {
        super.onLayout(z, i2, i3, i4, i5);
        if (z) {
            SearchAutoComplete searchAutoComplete = this.f140c;
            Rect rect = this.m;
            searchAutoComplete.getLocationInWindow(this.o);
            getLocationInWindow(this.p);
            int[] iArr = this.o;
            int i6 = iArr[1];
            int[] iArr2 = this.p;
            int i7 = i6 - iArr2[1];
            int i8 = iArr[0] - iArr2[0];
            rect.set(i8, i7, searchAutoComplete.getWidth() + i8, searchAutoComplete.getHeight() + i7);
            Rect rect2 = this.n;
            Rect rect3 = this.m;
            rect2.set(rect3.left, 0, rect3.right, i5 - i3);
            p pVar = this.l;
            if (pVar == null) {
                p pVar2 = new p(this.n, this.m, this.f140c);
                this.l = pVar2;
                setTouchDelegate(pVar2);
                return;
            }
            pVar.a(this.n, this.m);
        }
    }

    @Override // b.b.h.i0, android.view.View
    public void onMeasure(int i2, int i3) {
        int i4;
        if (this.D) {
            super.onMeasure(i2, i3);
            return;
        }
        int mode = View.MeasureSpec.getMode(i2);
        int size = View.MeasureSpec.getSize(i2);
        if (mode == Integer.MIN_VALUE) {
            int i5 = this.J;
            size = i5 > 0 ? Math.min(i5, size) : Math.min(getPreferredWidth(), size);
        } else if (mode == 0) {
            size = this.J;
            if (size <= 0) {
                size = getPreferredWidth();
            }
        } else if (mode == 1073741824 && (i4 = this.J) > 0) {
            size = Math.min(i4, size);
        }
        int mode2 = View.MeasureSpec.getMode(i3);
        int size2 = View.MeasureSpec.getSize(i3);
        if (mode2 == Integer.MIN_VALUE) {
            size2 = Math.min(getPreferredHeight(), size2);
        } else if (mode2 == 0) {
            size2 = getPreferredHeight();
        }
        super.onMeasure(View.MeasureSpec.makeMeasureSpec(size, 1073741824), View.MeasureSpec.makeMeasureSpec(size2, 1073741824));
    }

    @Override // android.view.View
    public void onRestoreInstanceState(Parcelable parcelable) {
        if (!(parcelable instanceof o)) {
            super.onRestoreInstanceState(parcelable);
            return;
        }
        o oVar = (o) parcelable;
        super.onRestoreInstanceState(oVar.getSuperState());
        t(oVar.f164b);
        requestLayout();
    }

    @Override // android.view.View
    public Parcelable onSaveInstanceState() {
        o oVar = new o(super.onSaveInstanceState());
        oVar.f164b = this.D;
        return oVar;
    }

    @Override // android.view.View
    public void onWindowFocusChanged(boolean z) {
        super.onWindowFocusChanged(z);
        post(this.R);
    }

    public void p() {
        int[] iArr = this.f140c.hasFocus() ? ViewGroup.FOCUSED_STATE_SET : ViewGroup.EMPTY_STATE_SET;
        Drawable background = this.f142e.getBackground();
        if (background != null) {
            background.setState(iArr);
        }
        Drawable background2 = this.f143f.getBackground();
        if (background2 != null) {
            background2.setState(iArr);
        }
        invalidate();
    }

    public final void q() {
        SpannableStringBuilder queryHint = getQueryHint();
        SearchAutoComplete searchAutoComplete = this.f140c;
        if (queryHint == null) {
            queryHint = "";
        }
        if (this.C && this.r != null) {
            int textSize = (int) (searchAutoComplete.getTextSize() * 1.25d);
            this.r.setBounds(0, 0, textSize, textSize);
            SpannableStringBuilder spannableStringBuilder = new SpannableStringBuilder("   ");
            spannableStringBuilder.setSpan(new ImageSpan(this.r), 1, 2, 33);
            spannableStringBuilder.append(queryHint);
            queryHint = spannableStringBuilder;
        }
        searchAutoComplete.setHint(queryHint);
    }

    public final void r() {
        int i2 = 0;
        if (!((this.F || this.K) && !this.D) || (this.f145h.getVisibility() != 0 && this.j.getVisibility() != 0)) {
            i2 = 8;
        }
        this.f143f.setVisibility(i2);
    }

    @Override // android.view.ViewGroup, android.view.View
    public boolean requestFocus(int i2, Rect rect) {
        if (!this.I && isFocusable()) {
            if (!this.D) {
                boolean requestFocus = this.f140c.requestFocus(i2, rect);
                if (requestFocus) {
                    t(false);
                }
                return requestFocus;
            }
            return super.requestFocus(i2, rect);
        }
        return false;
    }

    /* JADX WARN: Code restructure failed: missing block: B:16:0x001e, code lost:
        if (r2.K == false) goto L13;
     */
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public final void s(boolean z) {
        boolean z2 = this.F;
        int i2 = 0;
        if (z2) {
            if ((z2 || this.K) && !this.D) {
                if (hasFocus()) {
                    if (!z) {
                    }
                    this.f145h.setVisibility(i2);
                }
            }
        }
        i2 = 8;
        this.f145h.setVisibility(i2);
    }

    public void setAppSearchData(Bundle bundle) {
        this.Q = bundle;
    }

    public void setIconified(boolean z) {
        if (z) {
            h();
        } else {
            l();
        }
    }

    public void setIconifiedByDefault(boolean z) {
        if (this.C == z) {
            return;
        }
        this.C = z;
        t(z);
        q();
    }

    public void setImeOptions(int i2) {
        this.f140c.setImeOptions(i2);
    }

    public void setInputType(int i2) {
        this.f140c.setInputType(i2);
    }

    public void setMaxWidth(int i2) {
        this.J = i2;
        requestLayout();
    }

    public void setOnCloseListener(k kVar) {
        this.y = kVar;
    }

    public void setOnQueryTextFocusChangeListener(View.OnFocusChangeListener onFocusChangeListener) {
        this.z = onFocusChangeListener;
    }

    public void setOnQueryTextListener(l lVar) {
        this.x = lVar;
    }

    public void setOnSearchClickListener(View.OnClickListener onClickListener) {
        this.B = onClickListener;
    }

    public void setOnSuggestionListener(m mVar) {
        this.A = mVar;
    }

    public void setQueryHint(CharSequence charSequence) {
        this.G = charSequence;
        q();
    }

    public void setQueryRefinementEnabled(boolean z) {
        this.H = z;
        b.k.a.a aVar = this.E;
        if (aVar instanceof r0) {
            ((r0) aVar).s = z ? 2 : 1;
        }
    }

    /* JADX WARN: Code restructure failed: missing block: B:32:0x009c, code lost:
        if (getContext().getPackageManager().resolveActivity(r2, 65536) != null) goto L29;
     */
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public void setSearchableInfo(SearchableInfo searchableInfo) {
        this.P = searchableInfo;
        boolean z = true;
        Intent intent = null;
        if (searchableInfo != null) {
            this.f140c.setThreshold(searchableInfo.getSuggestThreshold());
            this.f140c.setImeOptions(this.P.getImeOptions());
            int inputType = this.P.getInputType();
            if ((inputType & 15) == 1) {
                inputType &= -65537;
                if (this.P.getSuggestAuthority() != null) {
                    inputType = inputType | 65536 | 524288;
                }
            }
            this.f140c.setInputType(inputType);
            b.k.a.a aVar = this.E;
            if (aVar != null) {
                aVar.b(null);
            }
            if (this.P.getSuggestAuthority() != null) {
                r0 r0Var = new r0(getContext(), this, this.P, this.T);
                this.E = r0Var;
                this.f140c.setAdapter(r0Var);
                ((r0) this.E).s = this.H ? 2 : 1;
            }
            q();
        }
        SearchableInfo searchableInfo2 = this.P;
        if (searchableInfo2 != null && searchableInfo2.getVoiceSearchEnabled()) {
            if (this.P.getVoiceSearchLaunchWebSearch()) {
                intent = this.u;
            } else if (this.P.getVoiceSearchLaunchRecognizer()) {
                intent = this.v;
            }
            if (intent != null) {
            }
        }
        z = false;
        this.K = z;
        if (z) {
            this.f140c.setPrivateImeOptions("nm");
        }
        t(this.D);
    }

    public void setSubmitButtonEnabled(boolean z) {
        this.F = z;
        t(this.D);
    }

    public void setSuggestionsAdapter(b.k.a.a aVar) {
        this.E = aVar;
        this.f140c.setAdapter(aVar);
    }

    public final void t(boolean z) {
        this.D = z;
        int i2 = 0;
        int i3 = z ? 0 : 8;
        boolean z2 = !TextUtils.isEmpty(this.f140c.getText());
        this.f144g.setVisibility(i3);
        s(z2);
        this.f141d.setVisibility(z ? 8 : 0);
        if (this.q.getDrawable() == null || this.C) {
            i2 = 8;
        }
        this.q.setVisibility(i2);
        o();
        u(!z2);
        r();
    }

    public final void u(boolean z) {
        int i2 = 8;
        if (this.K && !this.D && z) {
            this.f145h.setVisibility(8);
            i2 = 0;
        }
        this.j.setVisibility(i2);
    }

    public SearchView(Context context, AttributeSet attributeSet) {
        this(context, attributeSet, R.attr.searchViewStyle);
    }

    public SearchView(Context context, AttributeSet attributeSet, int i2) {
        super(context, attributeSet, i2);
        this.m = new Rect();
        this.n = new Rect();
        this.o = new int[2];
        this.p = new int[2];
        this.R = new b();
        this.S = new c();
        this.T = new WeakHashMap<>();
        f fVar = new f();
        this.U = fVar;
        this.V = new g();
        h hVar = new h();
        this.W = hVar;
        i iVar = new i();
        this.a0 = iVar;
        j jVar = new j();
        this.b0 = jVar;
        this.c0 = new a();
        TypedArray obtainStyledAttributes = context.obtainStyledAttributes(attributeSet, b.b.b.u, i2, 0);
        y0 y0Var = new y0(context, obtainStyledAttributes);
        LayoutInflater.from(context).inflate(y0Var.m(9, R.layout.abc_search_view), (ViewGroup) this, true);
        SearchAutoComplete searchAutoComplete = (SearchAutoComplete) findViewById(R.id.search_src_text);
        this.f140c = searchAutoComplete;
        searchAutoComplete.setSearchView(this);
        this.f141d = findViewById(R.id.search_edit_frame);
        View findViewById = findViewById(R.id.search_plate);
        this.f142e = findViewById;
        View findViewById2 = findViewById(R.id.submit_area);
        this.f143f = findViewById2;
        ImageView imageView = (ImageView) findViewById(R.id.search_button);
        this.f144g = imageView;
        ImageView imageView2 = (ImageView) findViewById(R.id.search_go_btn);
        this.f145h = imageView2;
        ImageView imageView3 = (ImageView) findViewById(R.id.search_close_btn);
        this.i = imageView3;
        ImageView imageView4 = (ImageView) findViewById(R.id.search_voice_btn);
        this.j = imageView4;
        ImageView imageView5 = (ImageView) findViewById(R.id.search_mag_icon);
        this.q = imageView5;
        Drawable g2 = y0Var.g(10);
        AtomicInteger atomicInteger = q.f2214a;
        findViewById.setBackground(g2);
        findViewById2.setBackground(y0Var.g(14));
        imageView.setImageDrawable(y0Var.g(13));
        imageView2.setImageDrawable(y0Var.g(7));
        imageView3.setImageDrawable(y0Var.g(4));
        imageView4.setImageDrawable(y0Var.g(16));
        imageView5.setImageDrawable(y0Var.g(13));
        this.r = y0Var.g(12);
        b.b.a.n(imageView, getResources().getString(R.string.abc_searchview_description_search));
        this.s = y0Var.m(15, R.layout.abc_search_dropdown_item_icons_2line);
        this.t = y0Var.m(5, 0);
        imageView.setOnClickListener(fVar);
        imageView3.setOnClickListener(fVar);
        imageView2.setOnClickListener(fVar);
        imageView4.setOnClickListener(fVar);
        searchAutoComplete.setOnClickListener(fVar);
        searchAutoComplete.addTextChangedListener(this.c0);
        searchAutoComplete.setOnEditorActionListener(hVar);
        searchAutoComplete.setOnItemClickListener(iVar);
        searchAutoComplete.setOnItemSelectedListener(jVar);
        searchAutoComplete.setOnKeyListener(this.V);
        searchAutoComplete.setOnFocusChangeListener(new d());
        setIconifiedByDefault(y0Var.a(8, true));
        int f2 = y0Var.f(1, -1);
        if (f2 != -1) {
            setMaxWidth(f2);
        }
        this.w = y0Var.o(6);
        this.G = y0Var.o(11);
        int j2 = y0Var.j(3, -1);
        if (j2 != -1) {
            setImeOptions(j2);
        }
        int j3 = y0Var.j(2, -1);
        if (j3 != -1) {
            setInputType(j3);
        }
        setFocusable(y0Var.a(0, true));
        obtainStyledAttributes.recycle();
        Intent intent = new Intent("android.speech.action.WEB_SEARCH");
        this.u = intent;
        intent.addFlags(268435456);
        intent.putExtra("android.speech.extra.LANGUAGE_MODEL", "web_search");
        Intent intent2 = new Intent("android.speech.action.RECOGNIZE_SPEECH");
        this.v = intent2;
        intent2.addFlags(268435456);
        View findViewById3 = findViewById(searchAutoComplete.getDropDownAnchor());
        this.k = findViewById3;
        if (findViewById3 != null) {
            findViewById3.addOnLayoutChangeListener(new e());
        }
        t(this.C);
        q();
    }
}