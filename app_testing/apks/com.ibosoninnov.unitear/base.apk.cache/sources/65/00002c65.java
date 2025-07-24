package com.google.thirdparty.publicsuffix;

import com.google.common.annotations.GwtCompatible;
import com.google.common.base.Joiner;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Lists;
import java.util.List;

@GwtCompatible
/* loaded from: classes2.dex */
public final class TrieParser {
    private static final Joiner PREFIX_JOINER = Joiner.on("");

    /* JADX WARN: Removed duplicated region for block: B:25:0x0053  */
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    private static int doParseTrieToBuilder(List<CharSequence> list, CharSequence charSequence, int i, ImmutableMap.Builder<String, PublicSuffixType> builder) {
        int length = charSequence.length();
        int i2 = i;
        char c2 = 0;
        while (i2 < length && (c2 = charSequence.charAt(i2)) != '&' && c2 != '?' && c2 != '!' && c2 != ':' && c2 != ',') {
            i2++;
        }
        list.add(0, reverse(charSequence.subSequence(i, i2)));
        if (c2 == '!' || c2 == '?' || c2 == ':' || c2 == ',') {
            String join = PREFIX_JOINER.join(list);
            if (join.length() > 0) {
                builder.put(join, PublicSuffixType.fromCode(c2));
            }
        }
        int i3 = i2 + 1;
        if (c2 != '?' && c2 != ',') {
            while (i3 < length) {
                i3 += doParseTrieToBuilder(list, charSequence, i3, builder);
                if (charSequence.charAt(i3) == '?' || charSequence.charAt(i3) == ',') {
                    i3++;
                    break;
                }
                while (i3 < length) {
                }
            }
        }
        list.remove(0);
        return i3 - i;
    }

    public static ImmutableMap<String, PublicSuffixType> parseTrie(CharSequence charSequence) {
        ImmutableMap.Builder builder = ImmutableMap.builder();
        int length = charSequence.length();
        int i = 0;
        while (i < length) {
            i += doParseTrieToBuilder(Lists.newLinkedList(), charSequence, i, builder);
        }
        return builder.build();
    }

    private static CharSequence reverse(CharSequence charSequence) {
        return new StringBuilder(charSequence).reverse();
    }
}