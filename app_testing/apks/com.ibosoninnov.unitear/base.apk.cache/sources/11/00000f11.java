package com.google.android.gms.common.server.response;

import android.util.Log;
import androidx.annotation.RecentlyNonNull;
import c.b.a.a.a;
import com.google.android.gms.common.annotation.KeepForSdk;
import com.google.android.gms.common.internal.ShowFirstParty;
import com.google.android.gms.common.server.response.FastJsonResponse;
import com.google.android.gms.common.util.Base64Utils;
import com.google.android.gms.common.util.JsonUtils;
import com.google.android.material.internal.StaticLayoutBuilderCompat;
import com.google.android.material.shadow.ShadowDrawableWrapper;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.math.BigDecimal;
import java.math.BigInteger;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.Stack;

/* compiled from: com.google.android.gms:play-services-base@@17.4.0 */
@ShowFirstParty
@KeepForSdk
/* loaded from: classes.dex */
public class FastParser<T extends FastJsonResponse> {
    private static final char[] zaf = {'u', 'l', 'l'};
    private static final char[] zag = {'r', 'u', 'e'};
    private static final char[] zah = {'r', 'u', 'e', '\"'};
    private static final char[] zai = {'a', 'l', 's', 'e'};
    private static final char[] zaj = {'a', 'l', 's', 'e', '\"'};
    private static final char[] zak = {'\n'};
    private static final zaa<Integer> zam = new com.google.android.gms.common.server.response.zaa();
    private static final zaa<Long> zan = new zac();
    private static final zaa<Float> zao = new zab();
    private static final zaa<Double> zap = new zae();
    private static final zaa<Boolean> zaq = new zad();
    private static final zaa<String> zar = new zag();
    private static final zaa<BigInteger> zas = new zaf();
    private static final zaa<BigDecimal> zat = new zah();
    private final char[] zaa = new char[1];
    private final char[] zab = new char[32];
    private final char[] zac = new char[1024];
    private final StringBuilder zad = new StringBuilder(32);
    private final StringBuilder zae = new StringBuilder(1024);
    private final Stack<Integer> zal = new Stack<>();

    /* compiled from: com.google.android.gms:play-services-base@@17.4.0 */
    @ShowFirstParty
    @KeepForSdk
    /* loaded from: classes.dex */
    public static class ParseException extends Exception {
        public ParseException(@RecentlyNonNull String str) {
            super(str);
        }

        public ParseException(@RecentlyNonNull String str, @RecentlyNonNull Throwable th) {
            super(str, th);
        }

        public ParseException(@RecentlyNonNull Throwable th) {
            super(th);
        }
    }

    /* compiled from: com.google.android.gms:play-services-base@@17.4.0 */
    /* loaded from: classes.dex */
    public interface zaa<O> {
        O zaa(FastParser fastParser, BufferedReader bufferedReader);
    }

    /* JADX DEBUG: Multi-variable search result rejected for r15v0, resolved type: com.google.android.gms.common.server.response.FastJsonResponse */
    /* JADX DEBUG: Type inference failed for r6v12. Raw type applied. Possible types: com.google.android.gms.common.server.response.FastParser$zaa<java.lang.Long>, com.google.android.gms.common.server.response.FastParser$zaa<O> */
    /* JADX DEBUG: Type inference failed for r6v16. Raw type applied. Possible types: com.google.android.gms.common.server.response.FastParser$zaa<java.lang.Float>, com.google.android.gms.common.server.response.FastParser$zaa<O> */
    /* JADX DEBUG: Type inference failed for r6v20. Raw type applied. Possible types: com.google.android.gms.common.server.response.FastParser$zaa<java.lang.Double>, com.google.android.gms.common.server.response.FastParser$zaa<O> */
    /* JADX DEBUG: Type inference failed for r6v24. Raw type applied. Possible types: com.google.android.gms.common.server.response.FastParser$zaa<java.math.BigDecimal>, com.google.android.gms.common.server.response.FastParser$zaa<O> */
    /* JADX DEBUG: Type inference failed for r6v29. Raw type applied. Possible types: com.google.android.gms.common.server.response.FastParser$zaa<java.lang.Boolean>, com.google.android.gms.common.server.response.FastParser$zaa<O> */
    /* JADX DEBUG: Type inference failed for r6v33. Raw type applied. Possible types: com.google.android.gms.common.server.response.FastParser$zaa<java.lang.String>, com.google.android.gms.common.server.response.FastParser$zaa<O> */
    /* JADX DEBUG: Type inference failed for r6v4. Raw type applied. Possible types: com.google.android.gms.common.server.response.FastParser$zaa<java.lang.Integer>, com.google.android.gms.common.server.response.FastParser$zaa<O> */
    /* JADX DEBUG: Type inference failed for r6v8. Raw type applied. Possible types: com.google.android.gms.common.server.response.FastParser$zaa<java.math.BigInteger>, com.google.android.gms.common.server.response.FastParser$zaa<O> */
    /* JADX WARN: Multi-variable type inference failed */
    private final boolean zaa(BufferedReader bufferedReader, FastJsonResponse fastJsonResponse) {
        HashMap hashMap;
        Map<String, FastJsonResponse.Field<?, ?>> fieldMappings = fastJsonResponse.getFieldMappings();
        String zaa2 = zaa(bufferedReader);
        if (zaa2 == null) {
            zaa(1);
            return false;
        }
        while (zaa2 != null) {
            FastJsonResponse.Field<?, ?> field = fieldMappings.get(zaa2);
            if (field == null) {
                zaa2 = zab(bufferedReader);
            } else {
                int i = 4;
                this.zal.push(4);
                switch (field.zaa) {
                    case 0:
                        if (field.zab) {
                            fastJsonResponse.zaa((FastJsonResponse.Field) field, (ArrayList<Integer>) zaa(bufferedReader, zam));
                        } else {
                            fastJsonResponse.zaa((FastJsonResponse.Field) field, zad(bufferedReader));
                        }
                        i = 4;
                        break;
                    case 1:
                        if (field.zab) {
                            fastJsonResponse.zab((FastJsonResponse.Field) field, (ArrayList<BigInteger>) zaa(bufferedReader, zas));
                        } else {
                            fastJsonResponse.zaa((FastJsonResponse.Field) field, zaf(bufferedReader));
                        }
                        i = 4;
                        break;
                    case 2:
                        if (field.zab) {
                            fastJsonResponse.zac(field, zaa(bufferedReader, zan));
                        } else {
                            fastJsonResponse.zaa((FastJsonResponse.Field) field, zae(bufferedReader));
                        }
                        i = 4;
                        break;
                    case 3:
                        if (field.zab) {
                            fastJsonResponse.zad(field, zaa(bufferedReader, zao));
                        } else {
                            fastJsonResponse.zaa((FastJsonResponse.Field) field, zag(bufferedReader));
                        }
                        i = 4;
                        break;
                    case 4:
                        if (field.zab) {
                            fastJsonResponse.zae(field, zaa(bufferedReader, zap));
                        } else {
                            fastJsonResponse.zaa(field, zah(bufferedReader));
                        }
                        i = 4;
                        break;
                    case 5:
                        if (field.zab) {
                            fastJsonResponse.zaf(field, zaa(bufferedReader, zat));
                        } else {
                            fastJsonResponse.zaa((FastJsonResponse.Field) field, zai(bufferedReader));
                        }
                        i = 4;
                        break;
                    case 6:
                        if (field.zab) {
                            fastJsonResponse.zag(field, zaa(bufferedReader, zaq));
                        } else {
                            fastJsonResponse.zaa(field, zaa(bufferedReader, false));
                        }
                        i = 4;
                        break;
                    case 7:
                        if (field.zab) {
                            fastJsonResponse.zah(field, zaa(bufferedReader, zar));
                        } else {
                            fastJsonResponse.zaa((FastJsonResponse.Field) field, zac(bufferedReader));
                        }
                        i = 4;
                        break;
                    case 8:
                        fastJsonResponse.zaa((FastJsonResponse.Field) field, Base64Utils.decode(zaa(bufferedReader, this.zac, this.zae, zak)));
                        i = 4;
                        break;
                    case 9:
                        fastJsonResponse.zaa((FastJsonResponse.Field) field, Base64Utils.decodeUrlSafe(zaa(bufferedReader, this.zac, this.zae, zak)));
                        i = 4;
                        break;
                    case 10:
                        char zaj2 = zaj(bufferedReader);
                        if (zaj2 == 'n') {
                            zab(bufferedReader, zaf);
                            hashMap = null;
                        } else if (zaj2 == '{') {
                            this.zal.push(1);
                            hashMap = new HashMap();
                            while (true) {
                                char zaj3 = zaj(bufferedReader);
                                if (zaj3 != 0) {
                                    if (zaj3 == '\"') {
                                        String zab = zab(bufferedReader, this.zab, this.zad, null);
                                        if (zaj(bufferedReader) != ':') {
                                            String valueOf = String.valueOf(zab);
                                            throw new ParseException(valueOf.length() != 0 ? "No map value found for key ".concat(valueOf) : new String("No map value found for key "));
                                        } else if (zaj(bufferedReader) != '\"') {
                                            String valueOf2 = String.valueOf(zab);
                                            throw new ParseException(valueOf2.length() != 0 ? "Expected String value for key ".concat(valueOf2) : new String("Expected String value for key "));
                                        } else {
                                            hashMap.put(zab, zab(bufferedReader, this.zab, this.zad, null));
                                            char zaj4 = zaj(bufferedReader);
                                            if (zaj4 != ',') {
                                                if (zaj4 == '}') {
                                                    zaa(1);
                                                } else {
                                                    StringBuilder sb = new StringBuilder(48);
                                                    sb.append("Unexpected character while parsing string map: ");
                                                    sb.append(zaj4);
                                                    throw new ParseException(sb.toString());
                                                }
                                            }
                                        }
                                    } else if (zaj3 == '}') {
                                        zaa(1);
                                    }
                                    i = 4;
                                    break;
                                } else {
                                    throw new ParseException("Unexpected EOF");
                                }
                            }
                        } else {
                            throw new ParseException("Expected start of a map object");
                        }
                        fastJsonResponse.zaa((FastJsonResponse.Field) field, (Map<String, String>) hashMap);
                        i = 4;
                    case 11:
                        if (field.zab) {
                            char zaj5 = zaj(bufferedReader);
                            if (zaj5 == 'n') {
                                zab(bufferedReader, zaf);
                                fastJsonResponse.addConcreteTypeArrayInternal(field, field.zae, null);
                                break;
                            } else {
                                this.zal.push(5);
                                if (zaj5 == '[') {
                                    fastJsonResponse.addConcreteTypeArrayInternal(field, field.zae, zaa(bufferedReader, field));
                                    break;
                                } else {
                                    throw new ParseException("Expected array start");
                                }
                            }
                        } else {
                            char zaj6 = zaj(bufferedReader);
                            if (zaj6 != 'n') {
                                this.zal.push(1);
                                if (zaj6 == '{') {
                                    try {
                                        FastJsonResponse zac = field.zac();
                                        zaa(bufferedReader, zac);
                                        fastJsonResponse.addConcreteTypeInternal(field, field.zae, zac);
                                        break;
                                    } catch (IllegalAccessException e2) {
                                        throw new ParseException("Error instantiating inner object", e2);
                                    } catch (InstantiationException e3) {
                                        throw new ParseException("Error instantiating inner object", e3);
                                    }
                                } else {
                                    throw new ParseException("Expected start of object");
                                }
                            } else {
                                zab(bufferedReader, zaf);
                                fastJsonResponse.addConcreteTypeInternal(field, field.zae, null);
                                break;
                            }
                        }
                    default:
                        throw new ParseException(a.g(30, "Invalid field type ", field.zaa));
                }
                zaa(i);
                zaa(2);
                char zaj7 = zaj(bufferedReader);
                if (zaj7 == ',') {
                    zaa2 = zaa(bufferedReader);
                } else if (zaj7 != '}') {
                    StringBuilder sb2 = new StringBuilder(55);
                    sb2.append("Expected end of object or field separator, but found: ");
                    sb2.append(zaj7);
                    throw new ParseException(sb2.toString());
                } else {
                    zaa2 = null;
                }
            }
        }
        zaa(1);
        return true;
    }

    private final String zab(BufferedReader bufferedReader) {
        bufferedReader.mark(1024);
        char zaj2 = zaj(bufferedReader);
        if (zaj2 == '\"') {
            if (bufferedReader.read(this.zaa) != -1) {
                char c2 = this.zaa[0];
                boolean z = false;
                do {
                    if (c2 != '\"' || z) {
                        z = c2 == '\\' ? !z : false;
                        if (bufferedReader.read(this.zaa) != -1) {
                            c2 = this.zaa[0];
                        } else {
                            throw new ParseException("Unexpected EOF while parsing string");
                        }
                    }
                } while (!Character.isISOControl(c2));
                throw new ParseException("Unexpected control character while reading string");
            }
            throw new ParseException("Unexpected EOF while parsing string");
        } else if (zaj2 == ',') {
            throw new ParseException("Missing value");
        } else {
            int i = 1;
            if (zaj2 == '[') {
                this.zal.push(5);
                bufferedReader.mark(32);
                if (zaj(bufferedReader) == ']') {
                    zaa(5);
                } else {
                    bufferedReader.reset();
                    boolean z2 = false;
                    boolean z3 = false;
                    while (i > 0) {
                        char zaj3 = zaj(bufferedReader);
                        if (zaj3 != 0) {
                            if (Character.isISOControl(zaj3)) {
                                throw new ParseException("Unexpected control character while reading array");
                            }
                            if (zaj3 == '\"' && !z2) {
                                z3 = !z3;
                            }
                            if (zaj3 == '[' && !z3) {
                                i++;
                            }
                            if (zaj3 == ']' && !z3) {
                                i--;
                            }
                            z2 = (zaj3 == '\\' && z3) ? !z2 : false;
                        } else {
                            throw new ParseException("Unexpected EOF while parsing array");
                        }
                    }
                    zaa(5);
                }
            } else if (zaj2 != '{') {
                bufferedReader.reset();
                zaa(bufferedReader, this.zac);
            } else {
                this.zal.push(1);
                bufferedReader.mark(32);
                char zaj4 = zaj(bufferedReader);
                if (zaj4 == '}') {
                    zaa(1);
                } else if (zaj4 == '\"') {
                    bufferedReader.reset();
                    zaa(bufferedReader);
                    do {
                    } while (zab(bufferedReader) != null);
                    zaa(1);
                } else {
                    StringBuilder sb = new StringBuilder(18);
                    sb.append("Unexpected token ");
                    sb.append(zaj4);
                    throw new ParseException(sb.toString());
                }
            }
        }
        char zaj5 = zaj(bufferedReader);
        if (zaj5 == ',') {
            zaa(2);
            return zaa(bufferedReader);
        } else if (zaj5 == '}') {
            zaa(2);
            return null;
        } else {
            StringBuilder sb2 = new StringBuilder(18);
            sb2.append("Unexpected token ");
            sb2.append(zaj5);
            throw new ParseException(sb2.toString());
        }
    }

    /* JADX INFO: Access modifiers changed from: private */
    public final String zac(BufferedReader bufferedReader) {
        return zaa(bufferedReader, this.zab, this.zad, null);
    }

    /* JADX INFO: Access modifiers changed from: private */
    public final int zad(BufferedReader bufferedReader) {
        int i;
        int i2;
        int zaa2 = zaa(bufferedReader, this.zac);
        int i3 = 0;
        if (zaa2 == 0) {
            return 0;
        }
        char[] cArr = this.zac;
        if (zaa2 > 0) {
            if (cArr[0] == '-') {
                i = Integer.MIN_VALUE;
                i2 = 1;
            } else {
                i = -2147483647;
                i2 = 0;
            }
            int i4 = i2;
            if (i2 < zaa2) {
                int i5 = i2 + 1;
                int digit = Character.digit(cArr[i2], 10);
                if (digit < 0) {
                    throw new ParseException("Unexpected non-digit character");
                }
                int i6 = -digit;
                i2 = i5;
                i3 = i6;
            }
            while (i2 < zaa2) {
                int i7 = i2 + 1;
                int digit2 = Character.digit(cArr[i2], 10);
                if (digit2 < 0) {
                    throw new ParseException("Unexpected non-digit character");
                }
                if (i3 < -214748364) {
                    throw new ParseException("Number too large");
                }
                int i8 = i3 * 10;
                if (i8 < i + digit2) {
                    throw new ParseException("Number too large");
                }
                i3 = i8 - digit2;
                i2 = i7;
            }
            if (i4 != 0) {
                if (i2 > 1) {
                    return i3;
                }
                throw new ParseException("No digits to parse");
            }
            return -i3;
        }
        throw new ParseException("No number to parse");
    }

    /* JADX INFO: Access modifiers changed from: private */
    public final long zae(BufferedReader bufferedReader) {
        long j;
        int zaa2 = zaa(bufferedReader, this.zac);
        long j2 = 0;
        if (zaa2 == 0) {
            return 0L;
        }
        char[] cArr = this.zac;
        if (zaa2 > 0) {
            int i = 0;
            if (cArr[0] == '-') {
                j = Long.MIN_VALUE;
                i = 1;
            } else {
                j = -9223372036854775807L;
            }
            int i2 = i;
            int i3 = 10;
            if (i < zaa2) {
                int i4 = i + 1;
                int digit = Character.digit(cArr[i], 10);
                if (digit < 0) {
                    throw new ParseException("Unexpected non-digit character");
                }
                i = i4;
                j2 = -digit;
            }
            while (i < zaa2) {
                int i5 = i + 1;
                int digit2 = Character.digit(cArr[i], i3);
                if (digit2 < 0) {
                    throw new ParseException("Unexpected non-digit character");
                }
                if (j2 < -922337203685477580L) {
                    throw new ParseException("Number too large");
                }
                long j3 = j2 * 10;
                long j4 = digit2;
                if (j3 < j + j4) {
                    throw new ParseException("Number too large");
                }
                j2 = j3 - j4;
                i = i5;
                i3 = 10;
            }
            if (i2 != 0) {
                if (i > 1) {
                    return j2;
                }
                throw new ParseException("No digits to parse");
            }
            return -j2;
        }
        throw new ParseException("No number to parse");
    }

    /* JADX INFO: Access modifiers changed from: private */
    public final BigInteger zaf(BufferedReader bufferedReader) {
        int zaa2 = zaa(bufferedReader, this.zac);
        if (zaa2 == 0) {
            return null;
        }
        return new BigInteger(new String(this.zac, 0, zaa2));
    }

    /* JADX INFO: Access modifiers changed from: private */
    public final float zag(BufferedReader bufferedReader) {
        int zaa2 = zaa(bufferedReader, this.zac);
        return zaa2 == 0 ? StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD : Float.parseFloat(new String(this.zac, 0, zaa2));
    }

    /* JADX INFO: Access modifiers changed from: private */
    public final double zah(BufferedReader bufferedReader) {
        int zaa2 = zaa(bufferedReader, this.zac);
        return zaa2 == 0 ? ShadowDrawableWrapper.COS_45 : Double.parseDouble(new String(this.zac, 0, zaa2));
    }

    /* JADX INFO: Access modifiers changed from: private */
    public final BigDecimal zai(BufferedReader bufferedReader) {
        int zaa2 = zaa(bufferedReader, this.zac);
        if (zaa2 == 0) {
            return null;
        }
        return new BigDecimal(new String(this.zac, 0, zaa2));
    }

    private final char zaj(BufferedReader bufferedReader) {
        if (bufferedReader.read(this.zaa) == -1) {
            return (char) 0;
        }
        while (Character.isWhitespace(this.zaa[0])) {
            if (bufferedReader.read(this.zaa) == -1) {
                return (char) 0;
            }
        }
        return this.zaa[0];
    }

    @KeepForSdk
    public void parse(@RecentlyNonNull InputStream inputStream, @RecentlyNonNull T t) {
        BufferedReader bufferedReader = new BufferedReader(new InputStreamReader(inputStream), 1024);
        try {
            try {
                this.zal.push(0);
                char zaj2 = zaj(bufferedReader);
                if (zaj2 != 0) {
                    if (zaj2 == '[') {
                        this.zal.push(5);
                        Map<String, FastJsonResponse.Field<?, ?>> fieldMappings = t.getFieldMappings();
                        if (fieldMappings.size() == 1) {
                            FastJsonResponse.Field<?, ?> value = fieldMappings.entrySet().iterator().next().getValue();
                            t.addConcreteTypeArrayInternal(value, value.zae, zaa(bufferedReader, value));
                        } else {
                            throw new ParseException("Object array response class must have a single Field");
                        }
                    } else if (zaj2 == '{') {
                        this.zal.push(1);
                        zaa(bufferedReader, t);
                    } else {
                        StringBuilder sb = new StringBuilder(19);
                        sb.append("Unexpected token: ");
                        sb.append(zaj2);
                        throw new ParseException(sb.toString());
                    }
                    zaa(0);
                    try {
                        bufferedReader.close();
                        return;
                    } catch (IOException unused) {
                        Log.w("FastParser", "Failed to close reader while parsing.");
                        return;
                    }
                }
                throw new ParseException("No data to parse");
            } catch (IOException e2) {
                throw new ParseException(e2);
            }
        } catch (Throwable th) {
            try {
                bufferedReader.close();
            } catch (IOException unused2) {
                Log.w("FastParser", "Failed to close reader while parsing.");
            }
            throw th;
        }
    }

    private static String zab(BufferedReader bufferedReader, char[] cArr, StringBuilder sb, char[] cArr2) {
        boolean z;
        sb.setLength(0);
        bufferedReader.mark(cArr.length);
        boolean z2 = false;
        boolean z3 = false;
        while (true) {
            int read = bufferedReader.read(cArr);
            if (read != -1) {
                for (int i = 0; i < read; i++) {
                    char c2 = cArr[i];
                    if (Character.isISOControl(c2)) {
                        if (cArr2 != null) {
                            for (char c3 : cArr2) {
                                if (c3 == c2) {
                                    z = true;
                                    break;
                                }
                            }
                        }
                        z = false;
                        if (!z) {
                            throw new ParseException("Unexpected control character while reading string");
                        }
                    }
                    if (c2 == '\"' && !z2) {
                        sb.append(cArr, 0, i);
                        bufferedReader.reset();
                        bufferedReader.skip(i + 1);
                        if (z3) {
                            return JsonUtils.unescapeString(sb.toString());
                        }
                        return sb.toString();
                    }
                    if (c2 == '\\') {
                        z2 = !z2;
                        z3 = true;
                    } else {
                        z2 = false;
                    }
                }
                sb.append(cArr, 0, read);
                bufferedReader.mark(cArr.length);
            } else {
                throw new ParseException("Unexpected EOF while parsing string");
            }
        }
    }

    private final void zab(BufferedReader bufferedReader, char[] cArr) {
        int i = 0;
        while (i < cArr.length) {
            int read = bufferedReader.read(this.zab, 0, cArr.length - i);
            if (read == -1) {
                throw new ParseException("Unexpected EOF");
            }
            for (int i2 = 0; i2 < read; i2++) {
                if (cArr[i2 + i] != this.zab[i2]) {
                    throw new ParseException("Unexpected character");
                }
            }
            i += read;
        }
    }

    private final String zaa(BufferedReader bufferedReader) {
        this.zal.push(2);
        char zaj2 = zaj(bufferedReader);
        if (zaj2 == '\"') {
            this.zal.push(3);
            String zab = zab(bufferedReader, this.zab, this.zad, null);
            zaa(3);
            if (zaj(bufferedReader) == ':') {
                return zab;
            }
            throw new ParseException("Expected key/value separator");
        } else if (zaj2 == ']') {
            zaa(2);
            zaa(1);
            zaa(5);
            return null;
        } else if (zaj2 == '}') {
            zaa(2);
            return null;
        } else {
            StringBuilder sb = new StringBuilder(19);
            sb.append("Unexpected token: ");
            sb.append(zaj2);
            throw new ParseException(sb.toString());
        }
    }

    private final <O> ArrayList<O> zaa(BufferedReader bufferedReader, zaa<O> zaaVar) {
        char zaj2 = zaj(bufferedReader);
        if (zaj2 == 'n') {
            zab(bufferedReader, zaf);
            return null;
        } else if (zaj2 == '[') {
            this.zal.push(5);
            ArrayList<O> arrayList = new ArrayList<>();
            while (true) {
                bufferedReader.mark(1024);
                char zaj3 = zaj(bufferedReader);
                if (zaj3 == 0) {
                    throw new ParseException("Unexpected EOF");
                }
                if (zaj3 != ',') {
                    if (zaj3 != ']') {
                        bufferedReader.reset();
                        arrayList.add(zaaVar.zaa(this, bufferedReader));
                    } else {
                        zaa(5);
                        return arrayList;
                    }
                }
            }
        } else {
            throw new ParseException("Expected start of array");
        }
    }

    private final String zaa(BufferedReader bufferedReader, char[] cArr, StringBuilder sb, char[] cArr2) {
        char zaj2 = zaj(bufferedReader);
        if (zaj2 != '\"') {
            if (zaj2 == 'n') {
                zab(bufferedReader, zaf);
                return null;
            }
            throw new ParseException("Expected string");
        }
        return zab(bufferedReader, cArr, sb, cArr2);
    }

    /* JADX INFO: Access modifiers changed from: private */
    public final boolean zaa(BufferedReader bufferedReader, boolean z) {
        while (true) {
            char zaj2 = zaj(bufferedReader);
            if (zaj2 != '\"') {
                if (zaj2 == 'f') {
                    zab(bufferedReader, z ? zaj : zai);
                    return false;
                } else if (zaj2 == 'n') {
                    zab(bufferedReader, zaf);
                    return false;
                } else if (zaj2 == 't') {
                    zab(bufferedReader, z ? zah : zag);
                    return true;
                } else {
                    StringBuilder sb = new StringBuilder(19);
                    sb.append("Unexpected token: ");
                    sb.append(zaj2);
                    throw new ParseException(sb.toString());
                }
            } else if (z) {
                throw new ParseException("No boolean value found in string");
            } else {
                z = true;
            }
        }
    }

    /* JADX DEBUG: Multi-variable search result rejected for r1v0, resolved type: java.util.ArrayList<T extends com.google.android.gms.common.server.response.FastJsonResponse> */
    /* JADX WARN: Multi-variable type inference failed */
    private final <T extends FastJsonResponse> ArrayList<T> zaa(BufferedReader bufferedReader, FastJsonResponse.Field<?, ?> field) {
        ArrayList<T> arrayList = (ArrayList<T>) new ArrayList();
        char zaj2 = zaj(bufferedReader);
        if (zaj2 == ']') {
            zaa(5);
            return arrayList;
        } else if (zaj2 == 'n') {
            zab(bufferedReader, zaf);
            zaa(5);
            return null;
        } else if (zaj2 == '{') {
            this.zal.push(1);
            while (true) {
                try {
                    FastJsonResponse zac = field.zac();
                    if (!zaa(bufferedReader, zac)) {
                        return arrayList;
                    }
                    arrayList.add(zac);
                    char zaj3 = zaj(bufferedReader);
                    if (zaj3 != ',') {
                        if (zaj3 == ']') {
                            zaa(5);
                            return arrayList;
                        }
                        StringBuilder sb = new StringBuilder(19);
                        sb.append("Unexpected token: ");
                        sb.append(zaj3);
                        throw new ParseException(sb.toString());
                    } else if (zaj(bufferedReader) == '{') {
                        this.zal.push(1);
                    } else {
                        throw new ParseException("Expected start of next object in array");
                    }
                } catch (IllegalAccessException e2) {
                    throw new ParseException("Error instantiating inner object", e2);
                } catch (InstantiationException e3) {
                    throw new ParseException("Error instantiating inner object", e3);
                }
            }
        } else {
            StringBuilder sb2 = new StringBuilder(19);
            sb2.append("Unexpected token: ");
            sb2.append(zaj2);
            throw new ParseException(sb2.toString());
        }
    }

    private final int zaa(BufferedReader bufferedReader, char[] cArr) {
        int i;
        char zaj2 = zaj(bufferedReader);
        if (zaj2 != 0) {
            if (zaj2 != ',') {
                if (zaj2 == 'n') {
                    zab(bufferedReader, zaf);
                    return 0;
                }
                bufferedReader.mark(1024);
                if (zaj2 == '\"') {
                    i = 0;
                    boolean z = false;
                    while (i < cArr.length && bufferedReader.read(cArr, i, 1) != -1) {
                        char c2 = cArr[i];
                        if (Character.isISOControl(c2)) {
                            throw new ParseException("Unexpected control character while reading string");
                        }
                        if (c2 == '\"' && !z) {
                            bufferedReader.reset();
                            bufferedReader.skip(i + 1);
                            return i;
                        }
                        z = c2 == '\\' ? !z : false;
                        i++;
                    }
                } else {
                    cArr[0] = zaj2;
                    i = 1;
                    while (i < cArr.length && bufferedReader.read(cArr, i, 1) != -1) {
                        if (cArr[i] == '}' || cArr[i] == ',' || Character.isWhitespace(cArr[i]) || cArr[i] == ']') {
                            bufferedReader.reset();
                            bufferedReader.skip(i - 1);
                            cArr[i] = 0;
                            return i;
                        }
                        i++;
                    }
                }
                if (i == cArr.length) {
                    throw new ParseException("Absurdly long value");
                }
                throw new ParseException("Unexpected EOF");
            }
            throw new ParseException("Missing value");
        }
        throw new ParseException("Unexpected EOF");
    }

    private final void zaa(int i) {
        if (!this.zal.isEmpty()) {
            int intValue = this.zal.pop().intValue();
            if (intValue != i) {
                throw new ParseException(a.h(46, "Expected state ", i, " but had ", intValue));
            }
            return;
        }
        StringBuilder sb = new StringBuilder(46);
        sb.append("Expected state ");
        sb.append(i);
        sb.append(" but had empty stack");
        throw new ParseException(sb.toString());
    }
}